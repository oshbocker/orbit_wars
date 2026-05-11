from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .config import EnvConfig, ImitationConfig, TrainConfig
from .env import OrbitWarsEnv, _default_make_fn
from .features import (
    GLOBAL_DIM,
    KNN_SCALAR_DIM,
    SOURCE_SCALAR_DIM,
    TARGET_SCALAR_DIM,
    FleetTransitState,
    SourceDecision,
    compute_fleet_transit,
    encode_source_decision,
    fleet_speed,
)
from .game_types import GameState, parse_observation
from .policy import TransformerPolicy


@dataclass
class DemonstrationBuffer:
    """Stores (state, action) pairs from expert demonstrations."""

    global_features: list[np.ndarray] = field(default_factory=list)
    source_scalars: list[np.ndarray] = field(default_factory=list)
    source_positions: list[np.ndarray] = field(default_factory=list)
    knn_scalars: list[np.ndarray] = field(default_factory=list)
    knn_positions: list[np.ndarray] = field(default_factory=list)
    target_scalars: list[np.ndarray] = field(default_factory=list)
    target_positions: list[np.ndarray] = field(default_factory=list)
    target_mask: list[np.ndarray] = field(default_factory=list)
    target_index: list[int] = field(default_factory=list)
    fraction_bin: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.global_features)

    def _to_tensors(self, indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "global_features": torch.from_numpy(
                np.array([self.global_features[i] for i in indices], dtype=np.float32)
            ).to(device),
            "source_scalars": torch.from_numpy(
                np.array([self.source_scalars[i] for i in indices], dtype=np.float32)
            ).to(device),
            "source_positions": torch.from_numpy(
                np.array([self.source_positions[i] for i in indices], dtype=np.float32)
            ).to(device),
            "knn_scalars": torch.from_numpy(
                np.array([self.knn_scalars[i] for i in indices], dtype=np.float32)
            ).to(device),
            "knn_positions": torch.from_numpy(
                np.array([self.knn_positions[i] for i in indices], dtype=np.float32)
            ).to(device),
            "target_scalars": torch.from_numpy(
                np.array([self.target_scalars[i] for i in indices], dtype=np.float32)
            ).to(device),
            "target_positions": torch.from_numpy(
                np.array([self.target_positions[i] for i in indices], dtype=np.float32)
            ).to(device),
            "target_mask": torch.from_numpy(
                np.array([self.target_mask[i] for i in indices])
            ).to(device).bool(),
            "target_index": torch.tensor(
                [self.target_index[i] for i in indices], dtype=torch.long, device=device
            ),
            "fraction_bin": torch.tensor(
                [self.fraction_bin[i] for i in indices], dtype=torch.long, device=device
            ),
        }

    def sample_batch(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        indices = np.random.randint(0, len(self), size=min(batch_size, len(self)))
        return self._to_tensors(indices, device)

    def sample_indices(self, indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
        return self._to_tensors(indices, device)

    def add(self, decision: SourceDecision, target_idx: int, frac_bin: int) -> None:
        self.global_features.append(decision.global_features)
        self.source_scalars.append(decision.source_scalars)
        self.source_positions.append(decision.source_position)
        self.knn_scalars.append(decision.knn_scalars)
        self.knn_positions.append(decision.knn_positions)
        self.target_scalars.append(decision.target_scalars)
        self.target_positions.append(decision.target_positions)
        self.target_mask.append(decision.target_mask)
        self.target_index.append(target_idx)
        self.fraction_bin.append(frac_bin)


def _map_to_action_space(
    expert_angle: float,
    expert_ships: int,
    src_ships: int,
    decision: SourceDecision,
    env_cfg: EnvConfig,
) -> tuple[int, int]:
    """Map an expert move (angle, ships) to our (target_index, fraction_bin).

    target_index: 0=NoOp, 1..T = target planets (offset by 1 for NoOp token)
    fraction_bin: index into env_cfg.ship_fractions
    """
    # Find closest target by angular difference (tolerance 90 degrees)
    best_idx = -1
    best_diff = math.radians(90)
    for i, tgt_angle in enumerate(decision.target_angles):
        diff = abs(_angle_diff(expert_angle, tgt_angle))
        if diff < best_diff:
            best_diff = diff
            best_idx = i

    if best_idx < 0:
        return 0, 0  # NoOp — no matching target

    # Map fraction
    frac = expert_ships / max(src_ships, 1)
    frac = max(0.0, min(1.0, frac))
    fractions = env_cfg.ship_fractions
    frac_bin = min(range(len(fractions)), key=lambda i: abs(fractions[i] - frac))

    # target_index is offset by 1 (0 = NoOp)
    return best_idx + 1, frac_bin


def _angle_diff(a: float, b: float) -> float:
    """Signed angular difference, wrapped to [-pi, pi]."""
    d = a - b
    return (d + math.pi) % (2 * math.pi) - math.pi


def _resolve_expert(name: str):
    """Import and return the expert agent function by name."""
    if name == "apex":
        from agents.apex import agent
        return agent
    if name == "hybrid":
        from agents.hybrid import agent
        return agent
    raise ValueError(f"Unknown BC expert: {name!r} (expected 'apex' or 'hybrid')")


def collect_demonstrations(
    n_games: int,
    cfg: TrainConfig,
    opponent_name: str = "random",
) -> DemonstrationBuffer:
    """Run expert agent for n_games and record (state, action) pairs.

    The expert plays as player 0 against the specified opponent.
    """
    expert_agent = _resolve_expert(cfg.imitation.bc_expert)
    from .opponents import build_opponent

    buffer = DemonstrationBuffer()
    opponent = build_opponent(opponent_name)
    make_fn = _default_make_fn()
    mapped = 0
    unmapped = 0

    for game_i in range(n_games):
        env = make_fn("orbit_wars", configuration={"seed": game_i + 100}, debug=False)
        env.reset(num_agents=2)
        states = env.step([[], []])
        done = False

        while not done:
            obs_p0 = states[0].observation if hasattr(states[0], "observation") else states[0]["observation"]
            obs_p1 = states[1].observation if hasattr(states[1], "observation") else states[1]["observation"]

            # Skip early-game steps (expert is intentionally passive)
            state = parse_observation(obs_p0)
            if state.step < cfg.imitation.bc_skip_steps:
                opp_moves = opponent.act(obs_p1)
                states = env.step([expert_agent(obs_p0) or [], opp_moves])
                status_p0 = states[0].status if hasattr(states[0], "status") else states[0]["status"]
                done = status_p0 != "ACTIVE"
                continue

            # Get expert agent's moves for player 0
            expert_moves = expert_agent(obs_p0)
            if not expert_moves:
                expert_moves = []

            # Build a lookup of expert moves by source planet id
            moves_by_src: dict[int, list[tuple[float, int]]] = {}
            for move in expert_moves:
                src_id = int(move[0])
                angle = float(move[1])
                ships = int(move[2])
                moves_by_src.setdefault(src_id, []).append((angle, ships))

            # Encode features like training does (state already parsed above)
            my_planets = sorted(
                [p for p in state.planets if p.owner == state.player],
                key=lambda p: -p.ships,
            )
            transit = compute_fleet_transit(state)

            for src in my_planets:
                decision = encode_source_decision(src, state, transit, cfg.env)
                src_moves = moves_by_src.get(src.id, [])

                if not src_moves:
                    # Expert chose NoOp for this planet
                    buffer.add(decision, target_idx=0, frac_bin=0)
                    unmapped += 1  # count as "no action" (still valid data)
                else:
                    # Take the first move for this source (largest fleet if multiple)
                    # Sum ships for all moves from this source
                    total_angle = src_moves[0][0]
                    total_ships = sum(s for _, s in src_moves)
                    tgt_idx, frac_bin = _map_to_action_space(
                        total_angle, total_ships, src.ships, decision, cfg.env,
                    )
                    # Verify target is valid in mask (avoid inf BC loss)
                    if tgt_idx > 0 and not decision.target_mask[tgt_idx + 1]:
                        tgt_idx = 0  # fall back to NoOp if target is masked
                    buffer.add(decision, target_idx=tgt_idx, frac_bin=frac_bin)
                    if tgt_idx > 0:
                        mapped += 1
                    else:
                        unmapped += 1

                    # Update transit for sequential consistency
                    if tgt_idx > 0:
                        target_offset = tgt_idx - 1
                        if target_offset < len(decision.target_planet_ids):
                            fraction = cfg.env.ship_fractions[frac_bin]
                            ships = int(src.ships * fraction)
                            if ships > 0:
                                target_id = decision.target_planet_ids[target_offset]
                                tgt_planet = state.planets_by_id.get(target_id)
                                if tgt_planet:
                                    speed = fleet_speed(ships)
                                    dist = math.hypot(src.x - tgt_planet.x, src.y - tgt_planet.y)
                                    eta = dist / max(speed, 0.1)
                                    transit.add_fleet(target_id, float(ships), eta, is_friendly=True)
                                src.ships = max(0, src.ships - ships)

            # Opponent acts
            opp_moves = opponent.act(obs_p1)

            states = env.step([list(expert_moves), opp_moves])
            status_p0 = states[0].status if hasattr(states[0], "status") else states[0]["status"]
            done = status_p0 != "ACTIVE"

        if (game_i + 1) % max(1, n_games // 5) == 0:
            print(f"  demo game {game_i + 1}/{n_games}  buffer={len(buffer)}")

    print(f"  Demo collection ({cfg.imitation.bc_expert}): {len(buffer)} samples, "
          f"{mapped} mapped, {unmapped} unmapped/noop")
    return buffer


def compute_bc_loss(
    policy: TransformerPolicy,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Cross-entropy loss on target selection + fraction selection."""
    outputs = policy(
        batch["global_features"],
        batch["source_scalars"],
        batch["source_positions"],
        batch["knn_scalars"],
        batch["knn_positions"],
        batch["target_scalars"],
        batch["target_positions"],
        batch["target_mask"],
    )

    target_index = batch["target_index"]
    fraction_bin = batch["fraction_bin"]

    # Clamp logits to avoid inf from masked positions
    safe_target_logits = outputs.target_logits.clamp(min=-1e4)

    # Target cross-entropy
    target_loss = F.cross_entropy(safe_target_logits, target_index)

    # Fraction cross-entropy (masked for NoOp entries)
    is_noop = target_index == 0
    if is_noop.all():
        return target_loss

    B = target_index.shape[0]
    frac_idx = (target_index - 1).clamp(min=0)
    frac_logits = outputs.fraction_logits[
        torch.arange(B, device=target_index.device), frac_idx
    ]  # [B, num_fractions]

    # Only compute fraction loss for non-NoOp entries
    frac_loss = F.cross_entropy(frac_logits[~is_noop], fraction_bin[~is_noop])

    return target_loss + frac_loss


def bc_pretrain(
    policy: TransformerPolicy,
    buffer: DemonstrationBuffer,
    imitation_cfg: ImitationConfig,
    device: torch.device,
    logger: object | None = None,
) -> None:
    """Supervised behavioral cloning pretraining."""
    N = len(buffer)
    if N == 0:
        print("  BC pretrain: empty buffer, skipping")
        return

    optimizer = torch.optim.Adam(policy.parameters(), lr=imitation_cfg.bc_lr)
    batch_size = imitation_cfg.bc_batch_size

    print(f"  BC pretrain: {imitation_cfg.bc_epochs} epochs, {N} samples, batch={batch_size}")

    for epoch in range(imitation_cfg.bc_epochs):
        order = np.random.permutation(N)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            indices = order[start:start + batch_size]
            batch = buffer.sample_indices(indices, device)

            loss = compute_bc_loss(policy, batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if logger is not None and hasattr(logger, "log_scalar"):
            logger.log_scalar("bc/loss", avg_loss, epoch)

        if (epoch + 1) % max(1, imitation_cfg.bc_epochs // 5) == 0:
            print(f"    epoch {epoch + 1}/{imitation_cfg.bc_epochs}  loss={avg_loss:.4f}")
