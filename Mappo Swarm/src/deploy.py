"""Deployment check for the trained actor (measured, not estimated).
Usage: python src/deploy.py --config configs/agent.yaml --run_dir runs/seed0
Exports the actor to TorchScript, verifies it matches the eager model on real observations,
and benchmarks (a) the policy forward pass and (b) the full decision step (policy + fuzzy filter +
loop guard) against deployment.latency_budget_ms. Writes <run_dir>/deployment_report.json."""
import argparse
import hashlib
import json
import os
import platform
import time

import numpy as np
import torch
import yaml

from env import SearchEnv
from mappo import MAPPO
from runtime import SwarmController, heuristic_action
from safety import FuzzySafetyFilter, LoopGuard


def stats_ms(v):
    v = np.asarray(v) * 1e3
    return {k: round(float(f(v)), 4) for k, f in
            (("mean", np.mean), ("p50", np.median), ("p95", lambda x: np.percentile(x, 95)),
             ("p99", lambda x: np.percentile(x, 99)), ("max", np.max))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/agent.yaml")
    ap.add_argument("--run_dir", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); d = cfg["deployment"]
    torch.set_num_threads(d["cpu_threads"])
    env = SearchEnv(cfg, 1000)
    algo = MAPPO(env.obs_dim, env.state_dim, env.n_actions, cfg["mappo"])
    ckpt = os.path.join(a.run_dir, "mappo.pt")
    algo.load(ckpt); algo.actor.eval()
    obs = env.reset(0)
    ts = torch.jit.trace(algo.actor, torch.as_tensor(obs))
    ts_path = os.path.join(a.run_dir, "actor_ts.pt"); ts.save(ts_path)
    ts = torch.jit.load(ts_path)

    # equivalence on observations from a real rollout (rule-based controller drives the swarm)
    diffs = []
    for _ in range(40):
        x = torch.as_tensor(env.observe())
        with torch.no_grad():
            diffs.append(float((algo.actor(x) - ts(x)).abs().max()))
        env.step(np.array([heuristic_action(env, i) for i in range(env.N)]))
    # latency
    env.reset(0)
    x = torch.as_tensor(env.observe())
    ctrl = SwarmController(algo, FuzzySafetyFilter(cfg, 0), cfg, "mappo")
    guard = LoopGuard(env.N, ctrl.limit)
    obs = env.observe()
    fwd, full = [], []
    with torch.no_grad():
        for _ in range(50):                                   # warm-up
            ts(x); ctrl.decide(env, obs, False, guard)
        for _ in range(d["bench_calls"]):
            t0 = time.perf_counter(); ts(x); fwd.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); ctrl.decide(env, obs, False, guard); full.append(time.perf_counter() - t0)
    rep = {"checkpoint_sha256": hashlib.sha256(open(ckpt, "rb").read()).hexdigest(),
           "actor_parameters": sum(p.numel() for p in algo.actor.parameters()),
           "critic_parameters_training_only": sum(p.numel() for p in algo.critic.parameters()),
           "actor_torchscript_kb": round(os.path.getsize(ts_path) / 1024, 1),
           "torchscript_max_abs_logit_diff": max(diffs),
           "forward_ms": stats_ms(fwd), "full_decision_ms": stats_ms(full),
           "latency_budget_ms": d["latency_budget_ms"], "swarm_size": env.N,
           "within_budget_p99": stats_ms(full)["p99"] <= d["latency_budget_ms"],
           "platform": platform.platform(), "python": platform.python_version(),
           "torch": torch.__version__, "threads": d["cpu_threads"]}
    json.dump(rep, open(os.path.join(a.run_dir, "deployment_report.json"), "w"), indent=2)
    print(json.dumps(rep, indent=2))
    assert rep["torchscript_max_abs_logit_diff"] < 1e-5, "TorchScript export does not match eager model"


if __name__ == "__main__":
    main()
