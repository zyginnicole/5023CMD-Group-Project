"""Train MAPPO. Usage: python src/train.py --config configs/agent.yaml --run_dir runs/seed0 --seed 0
train_policy / train_phase are reused by experiments.py (sweeps, forgetting)."""
import argparse
import copy
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
    for k in ("obs", "state", "act", "logp", "rew", "done", "val", "alive"):
        arrs = [np.asarray(e[k]) for e in buf]
        z = np.zeros((len(buf), T) + arrs[0].shape[1:], arrs[0].dtype)
        for j, a in enumerate(arrs):
            z[j, :len(a)] = a
        out[k] = z
    out["mask"] = np.zeros((len(buf), T), np.float32)
    for j, e in enumerate(buf):
        out["mask"][j, :len(e["rew"])] = 1
    return out


def train_phase(cfg, algo, ctrl, sources, n_up, rng, eval_sets, logf=None, verbose=True,
                eval_every=None, start=0):
    """Run n_up PPO updates.
    sources   : [(cfg_env, layout_seeds, probability)] - where training episodes come from
                (several sources = replay mixing, used by the forgetting experiment)
    eval_sets : {name: (cfg_env, layout_seeds)} evaluated every eval_every updates -> eval_<name>_found
    """
    T, tc = cfg["env"]["max_steps"], cfg["train"]
    eval_every = eval_every or tc["eval_every"]
    p = np.array([s[2] for s in sources], float); p /= p.sum()
    log = []
    for k in range(1, n_up + 1):
        u = start + k
        bufs, rets, found = [], [], []
        for _ in range(tc["episodes_per_update"]):
            noise = 1.0 + rng.random() * tc["noise_inject"]      # train-time noise injection
            src = sources[0] if len(sources) == 1 else sources[int(rng.choice(len(sources), p=p))]
            env = SearchEnv(src[0], int(rng.choice(src[1])), noise)
            st, buf = run_episode(env, ctrl, int(rng.integers(1 << 30)), stochastic=True, collect=True)
            bufs.append(buf); rets.append(st["ret"]); found.append(st["found_frac"])
        stats = algo.update(pad(bufs, T))
        rec = {"update": u, "train_return": float(np.mean(rets)), "train_found": float(np.mean(found)), **stats}
        if k % eval_every == 0 or k == n_up:
            for name, (c_env, lay) in eval_sets.items():
                rec[f"eval_{name}_found"] = evaluate(c_env, ctrl, lay, tc["eval_episodes"])["found_frac"]
        log.append(rec)
        if logf:
            logf.write(json.dumps(rec) + "\n"); logf.flush()
        if verbose and (k % 5 == 0 or k == n_up):
            print({a: round(v, 3) if isinstance(v, float) else v for a, v in rec.items()})
    return log


def build(cfg, seed):
    tr, ho = layouts(cfg)
    e0 = SearchEnv(cfg, tr[0])
    algo = MAPPO(e0.obs_dim, e0.state_dim, e0.n_actions, cfg["mappo"])
    ctrl = SwarmController(algo, FuzzySafetyFilter(cfg, seed), cfg, "mappo", seed)
    return e0, algo, ctrl, tr, ho


def train_policy(cfg, run, seed, n_up=None, verbose=True, plots=True, eval_every=None):
    """Full single-distribution training run. Returns (algo, ctrl, log)."""
    cfg = copy.deepcopy(cfg); cfg["seed"] = seed
    os.makedirs(run, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    e0, algo, ctrl, tr, ho = build(cfg, seed)
    n_up = n_up or cfg["train"]["n_updates"]
    yaml.safe_dump(cfg, open(os.path.join(run, "config_used.yaml"), "w"))   # exact config of this run
    t0 = time.perf_counter()
    with open(os.path.join(run, "train_log.jsonl"), "w") as logf:
        log = train_phase(cfg, algo, ctrl, [(cfg, tr, 1.0)], n_up, rng,
                          {"train": (cfg, tr), "heldout": (cfg, ho)}, logf, verbose, eval_every)
    algo.save(os.path.join(run, "mappo.pt"), dict(obs=e0.obs_dim, state=e0.state_dim))
    if plots:
        plot_training(run)
    secs = time.perf_counter() - t0
    update_timing(run, seed=seed, updates=n_up, train_seconds=round(secs, 1),
                  seconds_per_update=round(secs / n_up, 2))
    if verbose:
        print(f"Training wall-clock: {secs:.1f} s ({secs / 60:.1f} min) for {n_up} updates, "
              f"incl. periodic evaluation")
    return algo, ctrl, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/agent.yaml")
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    seed = cfg["seed"] if a.seed is None else a.seed
    train_policy(cfg, a.run_dir or cfg["logging"]["run_dir"], seed, a.updates)


if __name__ == "__main__":
    main()
