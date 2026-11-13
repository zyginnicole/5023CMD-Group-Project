"""Viva helper: prints the engine's inner workings from the REAL trained model.
Usage: python src/viva_inspect.py --config configs/agent.yaml --run_dir runs/seed0"""
import argparse
import os

import numpy as np
import torch
import yaml

from env import ACTIONS, SearchEnv, obs_groups
from mappo import MAPPO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/agent.yaml")
    ap.add_argument("--run_dir", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); m = cfg["mappo"]
    env = SearchEnv(cfg, 1000)
    algo = MAPPO(env.obs_dim, env.state_dim, env.n_actions, m)
    algo.load(os.path.join(a.run_dir, "mappo.pt"))
    n = lambda mod: sum(p.numel() for p in mod.parameters())
    print(f"== Architecture ==\nactor : {algo.actor}\n  params {n(algo.actor)} (deployed)\n"
          f"critic: {algo.critic}\n  params {n(algo.critic)} (training only, mode={'IPPO local' if algo.local else 'MAPPO central'})")
    print(f"\n== Observation vector per drone: {env.obs_dim} dims (patch k={env.k}) ==")
    for g, ix in obs_groups(env.k).items():
        print(f"  {g:22s} idx {ix[0]:>3d}..{ix[-1]:>3d}  ({len(ix)} dims)")
    print(f"critic state dim: {env.state_dim} = {env.N} x {env.obs_dim} obs + {2 * env.H * env.W} privileged (victims, obstacles)")
    print(f"actions: {dict(enumerate(ACTIONS))}")
    obs = env.reset(0)
    with torch.no_grad():
        lg = algo.actor(torch.as_tensor(obs))
    pr = torch.softmax(lg, -1)
    print("\n== One forward pass (episode start, per drone) ==")
    for i in range(env.N):
        print(f"  drone {i}: logits {np.round(lg[i].numpy(), 2)} -> probs {np.round(pr[i].numpy(), 3)} -> greedy '{ACTIONS[int(pr[i].argmax())]}'")
    print("\n== GAE on a 3-step toy episode (gamma={}, lambda={}) ==".format(m["gamma"], m["gae_lambda"]))
    rew, val, done = np.array([[1.0, 0.0, 2.0]]), np.array([[0.5, 0.4, 0.3]]), np.array([[0.0, 0.0, 1.0]])
    adv, ret = algo.gae(rew, val, done)
    g, l = m["gamma"], m["gae_lambda"]
    d2 = 2.0 - 0.3; d1 = 0.0 + g * 0.3 - 0.4; d0 = 1.0 + g * 0.4 - 0.5
    a2 = d2; a1 = d1 + g * l * a2; a0 = d0 + g * l * a1
    print(f"  code : adv {np.round(adv[0], 4)}\n  hand : adv {np.round([a0, a1, a2], 4)}")
    print("\n== PPO clipping (eps={}) ==".format(m["clip_eps"]))
    for ratio, A in ((1.5, 1.0), (1.5, -1.0), (0.5, 1.0), (0.5, -1.0)):
        s = min(ratio * A, np.clip(ratio, 1 - m["clip_eps"], 1 + m["clip_eps"]) * A)
        print(f"  ratio {ratio}, advantage {A:+.0f} -> surrogate {s:+.2f} ({'clipped' if s != ratio * A else 'unclipped'})")
    print("\n== Hyperparameters (evidence: experiments.py sweep / forgetting) ==")
    for k, v in m.items():
        print(f"  mappo.{k}: {v}")
    print(f"  reward: {cfg['reward']}")


if __name__ == "__main__":
    main()
