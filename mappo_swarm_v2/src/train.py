"""Train MAPPO. Usage: python src/train.py --config configs/agent.yaml [--updates N]"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import yaml

from env import SearchEnv
from evaluate import evaluate, layouts, plot_training
from mappo import MAPPO
from runtime import SwarmController, run_episode
from safety import FuzzySafetyFilter
from timing import update_timing


def pad(buf, T):
    """Stack variable-length episodes into [E, T, ...] arrays with a validity mask."""
    out = {}
    for k in ("obs", "state", "act", "logp", "rew", "done", "val"):
        arrs = [np.asarray(e[k]) for e in buf]
        z = np.zeros((len(buf), T) + arrs[0].shape[1:], arrs[0].dtype)
        for j, a in enumerate(arrs):
            z[j, :len(a)] = a
        out[k] = z
    out["mask"] = np.zeros((len(buf), T), np.float32)
    for j, e in enumerate(buf):
        out["mask"][j, :len(e["rew"])] = 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/agent.yaml")
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.seed is not None:
        cfg["seed"] = a.seed
    run = a.run_dir or cfg["logging"]["run_dir"]
    os.makedirs(run, exist_ok=True)
    seed = cfg["seed"]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr, ho = layouts(cfg)
    e0 = SearchEnv(cfg, tr[0])
    algo = MAPPO(e0.obs_dim, e0.state_dim, e0.n_actions, cfg["mappo"])
    ctrl = SwarmController(algo, FuzzySafetyFilter(cfg, seed), cfg, "mappo", seed)
    T, tc = cfg["env"]["max_steps"], cfg["train"]
    n_up = a.updates or tc["n_updates"]
    logf = open(os.path.join(run, "train_log.jsonl"), "w")
    t0 = time.perf_counter()
    for u in range(1, n_up + 1):
        bufs, rets, found = [], [], []
        for e in range(tc["episodes_per_update"]):
            noise = 1.0 + rng.random() * tc["noise_inject"]      # train-time noise injection
            env = SearchEnv(cfg, int(rng.choice(tr)), noise)
            st, buf = run_episode(env, ctrl, int(rng.integers(1 << 30)), stochastic=True, collect=True)
            bufs.append(buf); rets.append(st["ret"]); found.append(st["found_frac"])
        stats = algo.update(pad(bufs, T))
        rec = {"update": u, "train_return": float(np.mean(rets)), "train_found": float(np.mean(found)), **stats}
        if u % tc["eval_every"] == 0 or u == n_up:
            rec["eval_train_found"] = evaluate(cfg, ctrl, tr, tc["eval_episodes"])["found_frac"]
            rec["eval_heldout_found"] = evaluate(cfg, ctrl, ho, tc["eval_episodes"])["found_frac"]
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if u % 5 == 0 or u == n_up:
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in rec.items()})
    algo.save(os.path.join(run, "mappo.pt"), dict(obs=e0.obs_dim, state=e0.state_dim))
    plot_training(run)
    secs = time.perf_counter() - t0
    update_timing(run, seed=seed, updates=n_up, train_seconds=round(secs, 1),
                  seconds_per_update=round(secs / n_up, 2))
    print(f"Training wall-clock: {secs:.1f} s ({secs / 60:.1f} min) for {n_up} updates, incl. periodic evaluation")


if __name__ == "__main__":
    main()
