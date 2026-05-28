"""Profile env.step() breakdown."""
import time
import torch
from v2.config import load_v2_config
from v2.env import V2OrbitWarsEnv
from v2.features import encode_features
from v2.model import OrbitNet
from v2.actions import sample_actions, decode_sampled_actions
from src.game_types import parse_observation
from src.opponents import build_opponent
from v2.comet import comet_evacuation_moves
from v2.reward import compute_reward

cfg = load_v2_config("configs/v2_default.yaml")
device = torch.device("cpu")
model = OrbitNet(cfg.model).to(device)
opponent = build_opponent("apex")
env = V2OrbitWarsEnv(cfg, opponent, env_index=0)
feat = env.reset(seed=42)

t_comet = t_opp = t_kaggle = t_parse = t_reward = t_encode = 0
n_steps = 20

for step_i in range(n_steps):
    with torch.inference_mode():
        pf_t = torch.from_numpy(feat.planet_features).unsqueeze(0).to(device)
        gf_t = torch.from_numpy(feat.global_features).unsqueeze(0).to(device)
        pm_t = torch.from_numpy(feat.planet_mask).unsqueeze(0).to(device)
        om_t = torch.from_numpy(feat.own_mask).unsqueeze(0).to(device)
        rm_t = torch.from_numpy(feat.reachability_mask).unsqueeze(0).to(device)
        output = model(pf_t, gf_t, pm_t, om_t, rm_t)
        sampled = sample_actions(output, om_t, deterministic=False)
    state = env.last_state
    player_moves = decode_sampled_actions(sampled, output, feat, state, cfg.env)

    t0 = time.time()
    obs = env.last_obs
    comet_ids = None
    if hasattr(obs, "comet_planet_ids"):
        comet_ids = getattr(obs, "comet_planet_ids", None)
    elif isinstance(obs, dict):
        comet_ids = obs.get("comet_planet_ids")
    if comet_ids is not None:
        comet_ids = [int(x) for x in comet_ids]
    evac_moves, _ = comet_evacuation_moves(state, comet_ids, obs)
    all_moves = evac_moves + player_moves
    t_comet += time.time() - t0

    t0 = time.time()
    opp_moves = opponent.act(env.last_opp_obs[0])
    t_opp += time.time() - t0

    t0 = time.time()
    joint = [all_moves, opp_moves] if env.learner_player == 0 else [opp_moves, all_moves]
    states = env.env.step(joint)
    t_kaggle += time.time() - t0

    player_state = states[env.learner_player]
    new_obs = (
        player_state.observation
        if hasattr(player_state, "observation")
        else player_state.get("observation")
    )
    t0 = time.time()
    new_state = parse_observation(new_obs)
    t_parse += time.time() - t0

    t0 = time.time()
    status = str(getattr(player_state, "status", "ACTIVE"))
    done = status != "ACTIVE"
    _ = compute_reward(state, new_state, new_state.player, done, 0.0, cfg.reward)
    t_reward += time.time() - t0

    t0 = time.time()
    feat = encode_features(new_state, cfg.env)
    t_encode += time.time() - t0

    env.last_obs = new_obs
    env.last_opp_obs = [
        states[i].observation
        if hasattr(states[i], "observation")
        else states[i].get("observation")
        for i in range(2)
        if i != env.learner_player
    ]
    env.prev_state = state
    env.last_state = new_state

total = t_comet + t_opp + t_kaggle + t_parse + t_reward + t_encode
print(f"Over {n_steps} steps (per-step avg):")
print(f"  Comet evac:      {t_comet/n_steps*1000:6.1f}ms ({t_comet/total*100:4.1f}%)")
print(f"  Opponent (apex): {t_opp/n_steps*1000:6.1f}ms ({t_opp/total*100:4.1f}%)")
print(f"  Kaggle step:     {t_kaggle/n_steps*1000:6.1f}ms ({t_kaggle/total*100:4.1f}%)")
print(f"  Parse obs:       {t_parse/n_steps*1000:6.1f}ms ({t_parse/total*100:4.1f}%)")
print(f"  Compute reward:  {t_reward/n_steps*1000:6.1f}ms ({t_reward/total*100:4.1f}%)")
print(f"  Encode features: {t_encode/n_steps*1000:6.1f}ms ({t_encode/total*100:4.1f}%)")
print(f"  Total env.step:  {total/n_steps*1000:6.1f}ms")

# Also profile encode_features internals
from v2.state import compute_incoming_fleets
from v2.features import PLANET_FEAT_DIM, GLOBAL_FEAT_DIM
from src.features import passes_through_sun
import numpy as np

t_incoming = t_mask = t_features = 0
for _ in range(n_steps):
    st = env.last_state
    t0 = time.time()
    incoming = compute_incoming_fleets(st, st.player)
    t_incoming += time.time() - t0

    t0 = time.time()
    P = cfg.env.max_planets
    ps = [None] * P
    for p in st.planets:
        if 0 <= p.id < P:
            ps[p.id] = p
    rm = np.zeros((P, P), dtype=bool)
    for i in range(P):
        if ps[i] is None:
            continue
        for j in range(P):
            if i == j or ps[j] is None:
                continue
            if not passes_through_sun(ps[i].x, ps[i].y, ps[j].x, ps[j].y):
                rm[i, j] = True
    t_mask += time.time() - t0

print(f"\nEncode sub-timings (per step):")
print(f"  compute_incoming_fleets: {t_incoming/n_steps*1000:.1f}ms")
print(f"  reachability_mask:       {t_mask/n_steps*1000:.1f}ms")
