"""Behavioral cloning for V2 OrbitNet pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.game_types import parse_observation

from .config import V2Config, V2EnvConfig, V2ImitationConfig
from .features import V2Features, encode_features
from .model import OrbitNet


@dataclass
class V2DemonstrationBuffer:
    """Stores per-step (features, target_indices) from expert games.

    Each entry is one game step with full V2 feature tensors and
    the expert's target choice for each planet slot.
    """

    planet_features: list[np.ndarray]  # each [P, 22]
    global_features: list[np.ndarray]  # each [8]
    planet_mask: list[np.ndarray]  # each [P] bool
    own_mask: list[np.ndarray]  # each [P] bool
    reachability_mask: list[np.ndarray]  # each [P, P] bool
    target_indices: list[np.ndarray]  # each [P] int64 (0=hold, 1..P=target+1)
    frac_bins: list[np.ndarray]  # each [P] int64 (expert ship-fraction bin)
    supervise_mask: list[np.ndarray]  # each [P] bool (planets that contribute to BC loss)

    def __init__(self) -> None:
        self.planet_features = []
        self.global_features = []
        self.planet_mask = []
        self.own_mask = []
        self.reachability_mask = []
        self.target_indices = []
        self.frac_bins = []
        self.supervise_mask = []

    def __len__(self) -> int:
        return len(self.planet_features)

    def add(
        self,
        features: V2Features,
        targets: np.ndarray,
        frac_bins: np.ndarray,
        supervise_mask: np.ndarray,
    ) -> None:
        self.planet_features.append(features.planet_features.copy())
        self.global_features.append(features.global_features.copy())
        self.planet_mask.append(features.planet_mask.copy())
        self.own_mask.append(features.own_mask.copy())
        self.reachability_mask.append(features.reachability_mask.copy())
        self.target_indices.append(targets.copy())
        self.frac_bins.append(frac_bins.copy())
        self.supervise_mask.append(supervise_mask.copy())

    def sample_batch(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        indices = np.random.randint(0, len(self), size=min(batch_size, len(self)))
        return self._to_tensors(indices, device)

    def sample_indices(self, indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
        return self._to_tensors(indices, device)

    def _to_tensors(self, indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "planet_features": torch.from_numpy(
                np.array([self.planet_features[i] for i in indices], dtype=np.float32)
            ).to(device),
            "global_features": torch.from_numpy(
                np.array([self.global_features[i] for i in indices], dtype=np.float32)
            ).to(device),
            "planet_mask": torch.from_numpy(np.array([self.planet_mask[i] for i in indices]))
            .to(device)
            .bool(),
            "own_mask": torch.from_numpy(np.array([self.own_mask[i] for i in indices]))
            .to(device)
            .bool(),
            "reachability_mask": torch.from_numpy(
                np.array([self.reachability_mask[i] for i in indices])
            )
            .to(device)
            .bool(),
            "target_indices": torch.from_numpy(
                np.array([self.target_indices[i] for i in indices], dtype=np.int64)
            ).to(device),
            "frac_bins": torch.from_numpy(
                np.array([self.frac_bins[i] for i in indices], dtype=np.int64)
            ).to(device),
            # supervise_mask: fall back to own_mask for caches predating this field
            "supervise_mask": torch.from_numpy(
                np.array(
                    [
                        (
                            self.supervise_mask[i]
                            if getattr(self, "supervise_mask", None)
                            else self.own_mask[i]
                        )
                        for i in indices
                    ]
                )
            )
            .to(device)
            .bool(),
        }


# ── Helpers ──────────────────────────────────────────────────────────────────


def _angle_diff(a: float, b: float) -> float:
    """Signed angular difference, wrapped to [-pi, pi]."""
    d = a - b
    return (d + math.pi) % (2 * math.pi) - math.pi


def _frac_to_bin(frac: float, bins: list[float]) -> int:
    """Nearest discrete ship-fraction bin index for a continuous fraction."""
    return int(min(range(len(bins)), key=lambda k: abs(bins[k] - frac)))


def _resolve_expert(name: str):
    """Import and return the expert agent function by name (v5, producer, ...)."""
    from agents import load_named_agent

    return load_named_agent(name)


def _extract_obs(state_entry):
    """Extract observation dict from kaggle env state entry."""
    if hasattr(state_entry, "observation"):
        return state_entry.observation
    return state_entry["observation"]


def _extract_status(state_entry) -> str:
    """Extract status string from kaggle env state entry."""
    if hasattr(state_entry, "status"):
        return state_entry.status
    return state_entry["status"]


# ── Move Mapping ─────────────────────────────────────────────────────────────


def _map_expert_moves_to_v2(
    expert_moves: list,
    features: V2Features,
    env_cfg: V2EnvConfig,
    match_tolerance_deg: float = 90.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Map expert [src_id, angle, ships] moves to V2 target_indices [P].

    Returns (target_indices, frac_bins, supervise_mask, n_mapped, n_hold, n_dropped).
    target_indices[i] = 0 (hold) or j+1 (target planet slot j).
    frac_bins[i]      = expert ship-fraction bin index (0 for hold).
    supervise_mask[i] = whether this owned planet contributes to the BC loss.

    We supervise genuine holds (the expert launched nothing) and successful matches, but
    DROP expert launches we couldn't faithfully map (no angular match within
    tolerance, or matched target not reachable). Previously those were relabeled as
    "hold", which actively taught the clone to be passive.
    """
    P = env_cfg.max_planets
    target_indices = np.zeros(P, dtype=np.int64)
    frac_bins = np.zeros(P, dtype=np.int64)
    supervise_mask = features.own_mask.copy()
    n_mapped = 0
    n_hold = 0
    n_dropped = 0

    # Group moves by source planet id
    moves_by_src: dict[int, list[tuple[float, int]]] = {}
    for move in expert_moves:
        src_id = int(move[0])
        angle = float(move[1])
        ships = int(move[2])
        moves_by_src.setdefault(src_id, []).append((angle, ships))

    tol = math.radians(match_tolerance_deg)

    for i in range(P):
        if not features.own_mask[i]:
            continue
        src = features.planet_states[i]
        if src is None or src.ships <= 0:
            # Owns it but can't act -> hold is the correct label
            n_hold += 1
            continue

        src_moves = moves_by_src.get(src.id, [])
        if not src_moves:
            # Genuine hold: the expert launched nothing from this planet
            n_hold += 1
            continue

        # Expert launched: pick the largest-ships move as the primary target
        best_move = max(src_moves, key=lambda m: m[1])
        angle = best_move[0]
        total_ships = sum(s for _, s in src_moves)
        frac = total_ships / max(src.ships, 1)

        # Closest target by angular difference (within tolerance)
        best_j = -1
        best_diff = tol
        for j in range(P):
            if i == j:
                continue
            tgt = features.planet_states[j]
            if tgt is None:
                continue
            tgt_angle = math.atan2(tgt.y - src.y, tgt.x - src.x)
            diff = abs(_angle_diff(angle, tgt_angle))
            if diff < best_diff:
                best_diff = diff
                best_j = j

        # Apex acted but we can't faithfully map it -> DROP from supervision
        # (do NOT relabel as hold).
        if best_j < 0 or not features.reachability_mask[i, best_j]:
            supervise_mask[i] = False
            n_dropped += 1
            continue

        target_indices[i] = best_j + 1  # 0=hold, 1..P=targets
        frac_bins[i] = _frac_to_bin(max(0.2, min(1.0, frac)), env_cfg.ship_fractions)
        n_mapped += 1

    return target_indices, frac_bins, supervise_mask, n_mapped, n_hold, n_dropped


# ── Demo Collection ──────────────────────────────────────────────────────────


def collect_v2_demonstrations(
    n_games: int,
    cfg: V2Config,
    opponent_name: str = "random",
) -> V2DemonstrationBuffer:
    """Run expert agent for n_games, record V2 features + target indices per step."""
    from src.opponents import build_opponent

    expert_agent = _resolve_expert(cfg.imitation.bc_expert)
    opponent = build_opponent(opponent_name)
    buffer = V2DemonstrationBuffer()
    mapped_count = 0
    hold_count = 0
    dropped_count = 0

    for game_i in range(n_games):
        from kaggle_environments import make

        env = make("orbit_wars", configuration={"seed": game_i + 100}, debug=False)
        env.reset(num_agents=2)
        states = env.step([[], []])
        done = False

        while not done:
            obs_p0 = _extract_obs(states[0])
            obs_p1 = _extract_obs(states[1])

            state = parse_observation(obs_p0)

            # Skip early-game steps if configured
            if state.step < cfg.imitation.bc_skip_steps:
                expert_moves = expert_agent(obs_p0) or []
                opp_moves = opponent.act(obs_p1)
                states = env.step([list(expert_moves), opp_moves])
                done = _extract_status(states[0]) != "ACTIVE"
                continue

            # Get expert moves
            expert_moves = expert_agent(obs_p0) or []

            # Encode V2 features
            features = encode_features(state, cfg.env)

            # Map expert moves to V2 target indices + fraction bins
            targets, frac_bins, supervise_mask, n_mapped, n_hold, n_dropped = (
                _map_expert_moves_to_v2(
                    expert_moves,
                    features,
                    cfg.env,
                    match_tolerance_deg=cfg.imitation.bc_match_tolerance_deg,
                )
            )
            mapped_count += n_mapped
            hold_count += n_hold
            dropped_count += n_dropped

            # Only add if we own at least one planet
            if features.own_mask.any():
                buffer.add(features, targets, frac_bins, supervise_mask)

            # Step environment
            opp_moves = opponent.act(obs_p1)
            states = env.step([list(expert_moves), opp_moves])
            done = _extract_status(states[0]) != "ACTIVE"

        if (game_i + 1) % max(1, n_games // 5) == 0:
            print(f"  demo game {game_i + 1}/{n_games}  buffer={len(buffer)}")

    total = mapped_count + hold_count + dropped_count
    supervised = mapped_count + hold_count
    launches = mapped_count + dropped_count  # expert launches we tried to map
    capture = mapped_count / max(launches, 1) * 100  # of expert launches, how many mapped
    print(
        f"  V2 demo collection ({cfg.imitation.bc_expert}): {len(buffer)} samples, "
        f"{mapped_count} mapped sends, {hold_count} genuine holds, "
        f"{dropped_count} dropped (unmappable launches); "
        f"launch-capture={capture:.0f}%, supervised={supervised}/{total}"
    )
    return buffer


# ── BC Loss ──────────────────────────────────────────────────────────────────


def compute_v2_bc_loss(
    model: OrbitNet,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Cross-entropy on target selection + ship-fraction for owned planets.

    Target loss: CE(logits[i, :], target_indices[i]) over all owned planets.
    Fraction loss: CE(frac_logits[i, chosen_target, :], frac_bin[i]) over owned
    planets that send (target > 0). The fraction term lets BC clone the expert's
    fleet sizes, which the old target-only loss discarded.
    """
    output = model(
        batch["planet_features"],
        batch["global_features"],
        batch["planet_mask"],
        batch["own_mask"],
        batch.get("reachability_mask"),
    )

    logits = output.logits  # [B, P, P+1]
    targets = batch["target_indices"]  # [B, P]
    own_mask = batch["own_mask"]  # [B, P]
    # Supervise only genuine holds + mapped launches; drop unmappable expert launches
    # so we never teach spurious passivity.
    sup = own_mask & batch["supervise_mask"] if "supervise_mask" in batch else own_mask

    # Flatten to supervised owned planets only
    flat_logits = logits[sup]  # [N_sup, P+1]
    flat_targets = targets[sup]  # [N_sup]

    if flat_logits.shape[0] == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Clamp logits to avoid -inf from masked positions causing NaN
    safe_logits = flat_logits.clamp(min=-1e4)
    target_loss = F.cross_entropy(safe_logits, flat_targets)

    # Fraction loss on supervised planets that actually send (target > 0)
    frac_loss = torch.tensor(0.0, device=logits.device)
    send = sup & (targets > 0)  # [B, P]
    if send.any() and "frac_bins" in batch:
        bi = send.nonzero(as_tuple=False)  # [M, 2] -> (b, i)
        b_idx, i_idx = bi[:, 0], bi[:, 1]
        tslot = (targets[b_idx, i_idx] - 1).clamp(min=0)
        frac_rows = output.frac_logits[b_idx, i_idx, tslot]  # [M, K]
        frac_targets = batch["frac_bins"][b_idx, i_idx]  # [M]
        frac_loss = F.cross_entropy(frac_rows, frac_targets)

    return target_loss + frac_loss


# ── BC Pretraining ───────────────────────────────────────────────────────────


def v2_bc_pretrain(
    model: OrbitNet,
    buffer: V2DemonstrationBuffer,
    imitation_cfg: V2ImitationConfig,
    device: torch.device,
    logger: object | None = None,
) -> None:
    """Supervised BC pretraining for V2 OrbitNet."""
    N = len(buffer)
    if N == 0:
        print("  V2 BC pretrain: empty buffer, skipping")
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=imitation_cfg.bc_lr)
    batch_size = imitation_cfg.bc_batch_size

    print(f"  V2 BC pretrain: {imitation_cfg.bc_epochs} epochs, {N} samples, batch={batch_size}")

    model.train()
    for epoch in range(imitation_cfg.bc_epochs):
        order = np.random.permutation(N)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            indices = order[start : start + batch_size]
            batch = buffer.sample_indices(indices, device)

            loss = compute_v2_bc_loss(model, batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if logger is not None and hasattr(logger, "log_scalar"):
            logger.log_scalar("bc/loss", avg_loss, epoch)

        if (epoch + 1) % max(1, imitation_cfg.bc_epochs // 5) == 0:
            print(f"    epoch {epoch + 1}/{imitation_cfg.bc_epochs}  loss={avg_loss:.4f}")

    model.eval()
