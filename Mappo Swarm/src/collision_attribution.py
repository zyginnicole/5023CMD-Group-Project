"""Attribute every blocked move ("collision") to the action that caused it.

Usage (from the project folder):
  python src/collision_attribution.py --config configs/agent.yaml --run_dir runs/seed0 --seed 0
Without --run_dir only the heuristic and random controllers are checked (no PyTorch needed).

Method: before each env.step() the script asks, for every live drone, whether its EXECUTED
action would be a blocked move, using the same rules as SearchEnv.step (advance: the single
primary move is not valid; return away from base: no move towards base is valid). The
per-step prediction is asserted equal to env.step()'s own collision count, so the
attribution is exact, not an estimate. Episodes use the same layouts and episode seeds as
evaluate.py (held-out layouts, seeds 50000+e, greedy actions, no approval gate).
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import yaml

from env import ACTIONS, ADV, MOVES, RETURN, SearchEnv
from runtime import SwarmController
from safety import FuzzySafetyFilter, LoopGuard


def would_block(env, i, a):
    y, x = (int(v) for v in env.pos[i])
    if a == ADV:
        p = env._toward((y, x), env.goal[i])[:1]
        return bool(p) and not env._valid(y + MOVES[p[0]][0], x + MOVES[p[0]][1])
    if a == RETURN and not env.at_base(i):
        c = env._toward((y, x), env.base)
        return not any(env._valid(y + MOVES[m][0], x + MOVES[m][1]) for m in c)
    return False


def attribute(cfg, ctrl, seeds, n, use_filter):
    src, env_total = defaultdict(float), 0.0
    for e in range(n):
        env = SearchEnv(cfg, seeds[e % len(seeds)], 1.0)
        obs = env.reset(50_000 + e)
        guard = LoopGuard(env.N, ctrl.limit, intervene=True)
        for _ in range(cfg["env"]["max_steps"]):
            dec = ctrl.decide(env, obs, False, guard, use_filter)
            exe, pred = dec["exe"], 0
            for i in range(env.N):
                if not env.dead[i] and would_block(env, i, int(exe[i])):
                    src[ACTIONS[int(exe[i])]] += 1
                    pred += 1
            obs, _, done, info = env.step(exe, [fr.vetoed for fr in dec["filt"]])
            assert pred == info["collisions"], f"attribution mismatch {pred} != {info['collisions']}"
            env_total += info["collisions"]
            if done:
                break
    return {k: v / n for k, v in src.items()}, env_total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/agent.yaml")
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.seed is not None:
        cfg["seed"] = a.seed
    t = cfg["train"]
    tr = list(range(t["train_layout_seeds"]))
    ho = list(range(1000, 1000 + t["heldout_layout_seeds"]))
    n = a.episodes or cfg["eval"]["episodes"]
    filt = FuzzySafetyFilter(cfg, cfg["seed"] + 1)           # same seed as evaluate.py
    modes = {}
    if a.run_dir:
        from evaluate import evaluate                        # imports PyTorch lazily
        from mappo import MAPPO
        e0 = SearchEnv(cfg, tr[0])
        algo = MAPPO(e0.obs_dim, e0.state_dim, e0.n_actions, cfg["mappo"])
        algo.load(os.path.join(a.run_dir, "mappo.pt"))
        modes["mappo"] = SwarmController(algo, filt, cfg, "mappo")
        # evaluate.py ran MAPPO on the training layouts first, which advances the shared filter RNG.
        # Repeat that so the filter-on MAPPO row matches results.json exactly.
        evaluate(cfg, modes["mappo"], tr, n)
    modes["heuristic"] = SwarmController(None, filt, cfg, "heuristic")
    modes["random"] = SwarmController(None, filt, cfg, "random")
    print(f"{n} held-out episodes per row; blocked moves per episode by executed action")
    print(f"{'controller':10s} {'filter':6s} {'advance':>8s} {'return':>8s} {'other':>6s} {'total':>7s}")
    for name, ctrl in modes.items():
        for use_filter in (True, False):
            src, total = attribute(cfg, ctrl, ho, n, use_filter)
            other = total - src.get("advance", 0) - src.get("return", 0)
            print(f"{name:10s} {'on' if use_filter else 'off':6s} {src.get('advance', 0):8.1f} "
                  f"{src.get('return', 0):8.1f} {other:6.1f} {total:7.1f}")


if __name__ == "__main__":
    main()
