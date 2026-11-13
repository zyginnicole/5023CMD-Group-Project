"""Evaluation, baselines, ablations, XAI demo and plots (screenshot sources)."""
import argparse
import json
import logging
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from env import ACTIONS, SearchEnv
from mappo import MAPPO
from runtime import SwarmController, run_episode
from safety import FuzzySafetyFilter
from timing import update_timing
from xai import ApprovalGate, ScriptedSupervisor


def layouts(cfg):
    t = cfg["train"]
    return list(range(t["train_layout_seeds"])), list(range(1000, 1000 + t["heldout_layout_seeds"]))


def evaluate(cfg, ctrl, seeds, n, noise=1.0, **kw):
    rows = []
    for e in range(n):
        env = SearchEnv(cfg, seeds[e % len(seeds)], noise)
        st, _ = run_episode(env, ctrl, 50_000 + e, stochastic=False, **kw)
        rows.append(st)
    keys = ["ret", "found_frac", "loops", "vetoes", "false_pos", "collisions", "dead", "home", "return_at_base", "return_moving"]
    out = {k: float(np.mean([r.get(k, 0.0) for r in rows])) for k in keys}
    out.update({k + "_sd": float(np.std([r.get(k, 0.0) for r in rows])) for k in keys})   # across episodes
    out["actions"] = np.sum([r["actions"] for r in rows], 0).tolist()
    return out


def plot_training(run_dir):
    log = [json.loads(l) for l in open(os.path.join(run_dir, "train_log.jsonl"))]
    u = [l["update"] for l in log]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].plot(u, [l["train_return"] for l in log]); ax[0, 0].set_title("Mean training return")
    ev = [l for l in log if "eval_train_found" in l]
    ax[0, 1].plot([l["update"] for l in ev], [l["eval_train_found"] for l in ev], label="train layouts")
    ax[0, 1].plot([l["update"] for l in ev], [l["eval_heldout_found"] for l in ev], label="held-out layouts")
    ax[0, 1].set_title("Victims confirmed (fraction): generalisation gap"); ax[0, 1].legend()
    ax[1, 0].plot(u, [l["value_loss"] for l in log], label="value loss")
    ax[1, 0].plot(u, [l["policy_loss"] for l in log], label="policy loss"); ax[1, 0].legend()
    ax[1, 0].set_title("Losses")
    ax[1, 1].plot(u, [l["entropy"] for l in log], label="entropy")
    ax[1, 1].plot(u, [l["approx_kl"] for l in log], label="approx KL")
    ax[1, 1].plot(u, [l["clip_frac"] for l in log], label="clip fraction"); ax[1, 1].legend()
    ax[1, 1].set_title("Entropy / KL / clip fraction")
    for a in ax.ravel():
        a.set_xlabel("PPO update")
    plt.tight_layout(); plt.savefig(os.path.join(run_dir, "fig_training_curves.png"), dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/agent.yaml")
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--xai-demo", action="store_true", help="log one supervised episode")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.seed is not None:
        cfg["seed"] = a.seed
    run = a.run_dir or cfg["logging"]["run_dir"]
    tr, ho = layouts(cfg)
    e0 = SearchEnv(cfg, tr[0])
    algo = MAPPO(e0.obs_dim, e0.state_dim, e0.n_actions, cfg["mappo"])
    algo.load(os.path.join(run, "mappo.pt"))
    filt = FuzzySafetyFilter(cfg, cfg["seed"] + 1)
    mk = lambda mode: SwarmController(algo, filt, cfg, mode)
    n, shift = cfg["eval"]["episodes"], cfg["eval"]["shifted_noise"]
    t0 = time.perf_counter()
    res = {
        "mappo_train_layouts": evaluate(cfg, mk("mappo"), tr, n),
        "mappo_heldout_layouts": evaluate(cfg, mk("mappo"), ho, n),
        "mappo_heldout_noise_shift": evaluate(cfg, mk("mappo"), ho, n, noise=shift),
        "ablation_no_safety_filter": evaluate(cfg, mk("mappo"), ho, n, use_filter=False),
        "ablation_no_loop_guard": evaluate(cfg, mk("mappo"), ho, n, use_guard=False),
        "baseline_random": evaluate(cfg, mk("random"), ho, n),
        "baseline_heuristic": evaluate(cfg, mk("heuristic"), ho, n),
    }
    json.dump(res, open(os.path.join(run, "results.json"), "w"), indent=2)
    es = time.perf_counter() - t0
    update_timing(run, eval_seconds=round(es, 1), eval_episodes_per_condition=n, eval_conditions=len(res))
    print(f"Evaluation wall-clock: {es:.1f} s for {len(res)} conditions x {n} episodes")
    names = list(res)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].bar(names, [res[k]["found_frac"] for k in names]); ax[0].set_title("Victims confirmed (fraction)")
    ax[1].bar(names, [res[k]["ret"] for k in names]); ax[1].set_title("Mean episode return")
    for x in ax:
        x.tick_params(axis="x", rotation=60)
    plt.tight_layout(); plt.savefig(os.path.join(run, "fig_eval_comparison.png"), dpi=130); plt.close()
    plt.figure(figsize=(6, 4))
    plt.bar(ACTIONS, res["mappo_heldout_layouts"]["actions"]); plt.title("Proposed-action distribution (held-out)")
    plt.tight_layout(); plt.savefig(os.path.join(run, "fig_action_distribution.png"), dpi=130); plt.close()
    gap = res["mappo_train_layouts"]["found_frac"] - res["mappo_heldout_layouts"]["found_frac"]
    print(f"Generalisation gap (train - held-out victims found): {gap:+.3f}")
    for k in names:
        print(f"{k:28s} found={res[k]['found_frac']:.2f} return={res[k]['ret']:.1f} "
              f"+-{res[k]['found_frac_sd']:.2f} loops={res[k]['loops']:.1f} vetoes={res[k]['vetoes']:.1f} "
              f"dead={res[k]['dead']:.2f} ret@base={res[k]['return_at_base']:.1f}")
    if a.xai_demo:                                   # supervised episode -> exec log + audit log
        for f in ("execution.log", "audit.jsonl"):
            p = os.path.join(run, f)
            if os.path.exists(p): os.remove(p)
        lg = logging.getLogger("exec"); lg.setLevel(getattr(logging, cfg["logging"]["level"], logging.INFO))
        h = logging.FileHandler(os.path.join(run, "execution.log"))
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s")); lg.addHandler(h)
        gate = ApprovalGate(cfg, ScriptedSupervisor(cfg["xai"]["supervisor_attention"], 3),
                            os.path.join(run, "audit.jsonl"), 4)
        run_episode(SearchEnv(cfg, ho[0]), mk("mappo"), 777, stochastic=False, gate=gate, logger=lg)
        print("XAI gate:", gate.stats, "automation-bias rate on decoys:", gate.automation_bias_rate(),
              "| expected from simulator input (1 - attention) =", round(1 - cfg["xai"]["supervisor_attention"], 2),
              "| NOTE: demonstrates the measurement, not a finding")


if __name__ == "__main__":
    main()
