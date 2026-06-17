"""Rich-representation BC (Track 1 de-risk): feed producer's EXACT projection as features.

Same exact-relabeled macro-action targets as scripts/macro_bc_train.py, but the net now also
receives producer's per-edge projection grid (Δnet candidate score, ETA, size, validity) as
OrbitNet ``pair_features`` — the information producer reasons over. Labels = producer's ATTACK
launches (reinforce/regroup is left to v5's exact regroup gradient at execution).

Decisive de-risk: V2-snapshot BC-of-producer capped at 3% (val launch-acc ~0.59). If the rich
grid lifts launch-acc and the gate jumps toward parity, representation poverty WAS the wall,
and we scale to top-tier replays. Run scripts/rich_bc_agent.py for the arena gate.

CONTRACT CHANGE (2026-06-15): ``RichSelector.forward`` now returns a per-edge DELTA over
producer's real Δnet (0 at init), and the agent adds it to the EXACT per-candidate score under
v5's real ROI gate (see LEADERBOARD_CLIMB_PLAN 06-15: the old z-scored residual + ROI-off
harness lost 95% even at the producer prior, confounding the gate). ⚠️ RETRAIN TODO before
the trained ckpt is meaningful: (1) ``outputs/checkpoints/rich_bc_v5/ckpt40.pt`` is STALE
(old forward/scale); (2) for train/inference consistency the loss should compute CE on
``real_score + delta`` — the dataset must be rebuilt to store the raw per-candidate Δnet
(currently only the z-scored copy rides in pair[...,SCORE_CH]).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.producer_features import ProducerFeatureExtractor  # noqa: E402
from src.game_types import parse_observation  # noqa: E402
from v2.config import load_v2_config  # noqa: E402
from v2.features import encode_features  # noqa: E402
from v2.model import OrbitNet  # noqa: E402

N_PAIR = 4  # edge channels: [valid, score_z, eta/20, size/100]
SCORE_CH = 1  # index of the z-scored producer Δnet channel in pair_features


class RichSelector(torch.nn.Module):
    """OrbitNet + a DIRECT residual path on producer's Δnet score. At init the pair MLP
    is zero so target_logit = alpha * producer_score (alpha=1) => the model STARTS as
    producer (argmax score) and learns corrections from data. This is the
    'keep producer's exact evaluation, learn the delta' inductive bias — and it stops the
    rich score feature from being buried in the zero-init pair head."""

    def __init__(self, cfg):
        super().__init__()
        self.net = OrbitNet(cfg)

    def forward(self, pf, gf, pm, om, pair):
        # Returns a per-(source,target) DELTA over producer's real Δnet score. The pair
        # head is zero-init (OrbitNet._init_output_heads) so this is 0 everywhere at init
        # => the selector returns producer's exact score unchanged => reproduces v5. The
        # z-scored Δnet rides in `pair` (SCORE_CH) as a NET INPUT FEATURE; the residual is
        # added to the *real* score in the agent/hook, NOT here (that earlier z-scored add
        # broke v5's absolute-ROI fire gate — see LEADERBOARD_CLIMB_PLAN 06-15).
        return self.net(pf, gf, pm, om, None, pair).logits   # [B,P,P+1]


def _obs(e):
    return e["observation"] if not hasattr(e, "observation") else e.observation


def _comet_args(obs):
    cids = obs.get("comet_planet_ids") if isinstance(obs, dict) else getattr(obs, "comet_planet_ids", None)
    cdata = obs.get("comets") if isinstance(obs, dict) else getattr(obs, "comets", None)
    cids = [int(x) for x in cids] if cids is not None else None
    return cids, cdata


def _pair_features(grid: dict, P: int) -> np.ndarray:
    """Dense [P, P, N_PAIR] edge features from the extractor grid."""
    valid = grid["valid"]                                  # [P,P] bool
    score = grid["score"].clone()                          # [P,P] (-inf where invalid)
    finite = torch.isfinite(score)
    pf = np.zeros((P, P, N_PAIR), dtype=np.float32)
    pf[:, :, 0] = valid.float().numpy()
    if bool(finite.any()):
        vals = score[finite]
        mu = float(vals.mean())
        sd = (float(vals.std(unbiased=False)) + 1e-6) if vals.numel() > 1 else 1.0
        score0 = torch.where(finite, score, torch.zeros_like(score))  # -inf -> 0 before arith
        sz = torch.where(finite, (score0 - mu) / sd, torch.zeros_like(score))
        pf[:, :, 1] = sz.numpy()
    pf[:, :, 2] = (grid["eta"] / 20.0).numpy()
    pf[:, :, 3] = (grid["size"] / 100.0).numpy()
    return np.nan_to_num(pf, nan=0.0, posinf=0.0, neginf=0.0)


def _owner_by_id(obs, P):
    own = -np.ones(P, dtype=np.int64)
    pls = obs["planets"] if isinstance(obs, dict) else obs.planets
    for p in pls:
        pid = int(p[0])
        if 0 <= pid < P:
            own[pid] = int(p[1])
    return own


def example_from(cfg, ext, P, obs, sink):
    """One training example from a (obs, sink) pair. Features = the rolling Δnet grid; labels
    (`ti`) = the EXPERT's greedy-fired ATTACK target per source (reinforce/regroup -> hold).
    The expert is whoever produced `sink` (v5 in BC; v5-relabel of learner states in DAgger).
    Player derived from obs so any seat's obs can be relabeled by a clean v5."""
    state = parse_observation(obs)
    cids, cdata = _comet_args(obs)
    feats = encode_features(state, cfg.env, comet_ids=cids, comets_data=cdata)
    grid = ext._densify(sink)
    pair = _pair_features(grid, P)
    score_grid = grid["score"].numpy().astype(np.float32)   # [P,P] real Δnet (-inf invalid)
    player = int(obs["player"]) if isinstance(obs, dict) else int(obs.player)
    owner = _owner_by_id(obs, P)
    ti = np.zeros(P, dtype=np.int64)
    sup = feats.own_mask.copy()
    fs, ft, fv = sink.get("fired_src"), sink.get("fired_tgt"), sink.get("fired_valid")
    if fs is not None:
        for k in range(fs.shape[0]):
            if not bool(fv[k]):
                continue
            i, j = int(fs[k]), int(ft[k])
            if 0 <= i < P and 0 <= j < P and i != j and owner[j] != player and bool(feats.own_mask[i]):
                ti[i] = j + 1
    return {"pf": feats.planet_features, "gf": feats.global_features,
            "pm": feats.planet_mask, "om": feats.own_mask,
            "pair": pair, "score": score_grid, "ti": ti, "sup": sup}


def capturing_agent(mod, store):
    """Wrap a v5 module so each live (rolling) decision appends (obs, sink) to `store`."""
    def agent(obs, config=None):
        mod._FEATURE_SINK = {}
        try:
            act = mod.agent(obs)
        finally:
            sink, mod._FEATURE_SINK = mod._FEATURE_SINK, None
        if sink:
            store.append((obs, sink))
        return act
    return agent


def build_dataset(cfg, n_games, seed, expert):
    """LIVE, agent-consistent, rolling capture: instrument a v5-vs-v5 game so each seat's grid
    features AND greedy-fired labels come from its OWN run (rolling cache, labels in-grid by
    construction). Avoids the per-obs-reset / mixed-agent divergence."""
    from kaggle_environments import make

    from scripts.producer_features import load_v5_module
    P = cfg.env.max_planets
    ext = ProducerFeatureExtractor(max_planets=P)   # reused only for _densify
    ex = []
    for g in range(n_games):
        modA, modB = load_v5_module(), load_v5_module()
        capA, capB = [], []
        env = make("orbit_wars", configuration={"randomSeed": seed + g})
        env.run([capturing_agent(modA, capA), capturing_agent(modB, capB)])
        for cap in (capA, capB):
            for obs, sink in cap:
                ex.append(example_from(cfg, ext, P, obs, sink))
        print(f"  game {g}: {len(capA)+len(capB)} states, {len(ex)} examples", flush=True)
    return ex


def _stack(ex, k, dev):
    return torch.from_numpy(np.stack([e[k] for e in ex])).to(dev)


def make_batch(ex, idx, dev):
    sub = [ex[i] for i in idx]
    return {
        "planet_features": _stack(sub, "pf", dev), "global_features": _stack(sub, "gf", dev),
        "planet_mask": _stack(sub, "pm", dev), "own_mask": _stack(sub, "om", dev),
        "pair_features": _stack(sub, "pair", dev), "score": _stack(sub, "score", dev),
        "target_indices": _stack(sub, "ti", dev), "supervise_mask": _stack(sub, "sup", dev),
    }


# v5/producer 2P roi_threshold. Hold-column baseline so training fire+select matches inference:
# the agent fires candidate (src,tgt) iff real_Δnet + delta > ROI, and ranks by that sum.
ROI_BASELINE = 1.5


def combine_logits(delta, score, roi=ROI_BASELINE):
    """Selection logits that MATCH inference (agent returns cand_score + delta, ROI-gated):
    target column j = delta_j + real Δnet[.,j] (-inf invalid); hold column = delta_0 + ROI.
    At init (delta=0) argmax = producer's exact fire+select decision => reproduces v5."""
    out = delta.clone()                          # [B,P,P+1]
    out[..., 1:] = out[..., 1:] + score          # [B,P,P] real Δnet baseline
    out[..., 0] = out[..., 0] + roi
    return out


def loss_fn(model, b, lw):
    delta = model(b["planet_features"], b["global_features"], b["planet_mask"],
                  b["own_mask"], b["pair_features"])
    logits = combine_logits(delta, b["score"])
    sup = b["own_mask"] & b["supervise_mask"]
    flat = logits[sup]
    tgt = b["target_indices"][sup]
    if flat.shape[0] > 0:
        keep = flat.gather(1, tgt[:, None]).squeeze(1) > -1e3
        flat, tgt = flat[keep], tgt[keep]
    if flat.shape[0] == 0:
        return torch.zeros((), device=logits.device, requires_grad=True)
    ce = F.cross_entropy(flat.clamp(min=-1e4), tgt, reduction="none")
    w = torch.where(tgt > 0, lw, 1.0)
    return (ce * w).sum() / w.sum()


@torch.no_grad()
def accuracy(model, ex, dev, bs=256):
    model.eval()
    na = ca = nl = cl = 0
    for s in range(0, len(ex), bs):
        b = make_batch(ex, range(s, min(s + bs, len(ex))), dev)
        delta = model(b["planet_features"], b["global_features"], b["planet_mask"],
                      b["own_mask"], b["pair_features"])
        logits = combine_logits(delta, b["score"])
        pred = logits.argmax(-1)
        sup = b["own_mask"] & b["supervise_mask"]
        tgt = b["target_indices"]
        corr = (pred == tgt) & sup
        launch = sup & (tgt > 0)
        na += int(sup.sum())
        ca += int(corr.sum())
        nl += int(launch.sum())
        cl += int((corr & launch).sum())
    return (ca / na if na else 0.0), (cl / nl if nl else 0.0), nl


def train_selector(cfg, ex, dev, *, epochs=25, lr=5e-4, clip=1.0, wd=1e-4,
                   launch_weight=5.0, batch=64, init_state=None, seed=0):
    """Train a RichSelector on examples `ex`; return (best_state_dict, best_val_acc).
    init_state warm-starts (DAgger iterations); best-val snapshot is kept because the
    producer-prior init already reproduces v5 (val acc starts ~parity and can only drift)."""
    import copy
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ex))
    nv = max(1, len(ex) // 10)
    val = [ex[i] for i in perm[:nv]]
    train = [ex[i] for i in perm[nv:]]
    print(f"train={len(train)} val={len(val)}  pair_feat_dim={N_PAIR}")

    model = RichSelector(cfg.model).to(dev)
    if init_state is not None:
        model.load_state_dict(init_state)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    acc0, lacc0, nl0 = accuracy(model, val, dev)
    print(f"epoch  -1  (init)  val acc={acc0:.3f}  launch_acc={lacc0:.3f}(n={nl0})", flush=True)
    best_acc, best_state = acc0, copy.deepcopy(model.state_dict())
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(train))
        tot = nb = 0
        for s in range(0, len(train), batch):
            b = make_batch(train, order[s:s + batch], dev)
            loss = loss_fn(model, b, launch_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        if ep % 2 == 0 or ep == epochs - 1:
            acc, lacc, nl = accuracy(model, val, dev)
            tag = ""
            if acc > best_acc:
                best_acc, best_state, tag = acc, copy.deepcopy(model.state_dict()), "  *best"
            print(f"epoch {ep:3d}  loss={tot/nb:.4f}  val acc={acc:.3f}  launch_acc={lacc:.3f}(n={nl}){tag}", flush=True)
    return best_state, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2_exit.yaml")
    ap.add_argument("--expert", default="producer")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--launch_weight", type=float, default=5.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--cache", default="outputs/macro_bc/rich_producer40.npz")
    ap.add_argument("--out", default="outputs/checkpoints/rich_bc_producer/ckpt.pt")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    cfg = load_v2_config(args.config)
    cfg.model.use_pair_features = True
    cfg.model.pair_feat_dim = N_PAIR
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = Path(args.cache)
    if cache.exists() and not args.rebuild:
        ex = list(np.load(cache, allow_pickle=True)["ex"])
        print(f"loaded {len(ex)} examples from {cache}")
    else:
        ex = build_dataset(cfg, args.games, args.seed, args.expert)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, ex=np.array(ex, dtype=object))
        print(f"saved {len(ex)} -> {cache}")

    best_state, best_acc = train_selector(
        cfg, ex, dev, epochs=args.epochs, lr=args.lr, clip=args.clip, wd=args.wd,
        launch_weight=args.launch_weight, batch=args.batch)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "config": args.config, "pair_feat_dim": N_PAIR}, out)
    print(f"saved best-val (acc={best_acc:.3f}) -> {out}")


if __name__ == "__main__":
    main()
