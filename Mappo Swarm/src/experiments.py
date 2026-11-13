"""Evidence for the Task 2 rationale. Every number the report cites for a design choice should come from here.

  python src/experiments.py sweep      --config configs/agent.yaml --out runs/sweep [--groups lr,epochs] [--seeds 0 1]
  python src/experiments.py forgetting --config configs/agent.yaml --out runs/forgetting
  python src/experiments.py attention  --config configs/agent.yaml --run_dir runs/seed0

sweep groups: base, lr, epochs, minibatch, gamma, pool, reward, model
  pool   : training-layout pool size -> overfitting (generalisation gap) evidence
  reward : each reward term zeroed in turn -> reward-structure justification
  model  : IPPO (local critic) and value-clip off -> model-choice justification
Per-variant results are cached in <out>/variants/, so an interrupted sweep resumes where it stopped.
Results are descriptive (few seeds, short budgets); no significance is claimed.
"""
import argparse
import copy
import json
import os
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from env import SearchEnv
from evaluate import evaluate, layouts
from runtime import SwarmController
from safety import FuzzySafetyFilter

METRICS = ["found_frac", "ret", "collisions", "dead", "loops"]


def set_path(cfg, path, value):
    d = cfg
    *head, last = path.split(".")
    for k in head:
        d = d[k]
    d[last] = value


def variants(cfg, groups):
    """name -> (group, {dotted.path: value}). 'base' is always included."""
    x, out = cfg["experiments"], {"base": ("base", {})}
    def add(group, grid, path, label):
        base_v = cfg
        for k in path.split("."):
            base_v = base_v[k]
        for v in grid:
            if v != base_v:
                out[f"{label}={v}"] = (group, {path: v})
    if "lr" in groups:        add("lr", x["lr"], "mappo.learning_rate", "lr")
    if "epochs" in groups:    add("epochs", x["ppo_epochs"], "mappo.ppo_epochs", "epochs")
    if "minibatch" in groups: add("minibatch", x["minibatch_size"], "mappo.minibatch_size", "minibatch")
    if "gamma" in groups:     add("gamma", x["gamma"], "mappo.gamma", "gamma")
    if "pool" in groups:      add("pool", x["layout_pool"], "train.train_layout_seeds", "layouts")
    if "reward" in groups:
        for t in x["reward_ablate"]:
            out[f"no_{t}_reward"] = ("reward", {f"reward.{t}": 0.0})
    if "model" in groups:
        out["IPPO_local_critic"] = ("model", {"mappo.critic": "local"})
        out["value_clip_off"] = ("model", {"mappo.value_clip": False})
    return out


def safe(name):
    return re.sub(r"[^A-Za-z0-9_.=-]", "_", name)


def run_variant(cfg, name, overrides, seed, updates, n_eval, out):
    from train import train_policy                      # torch imported lazily
    path = os.path.join(out, "variants", f"{safe(name)}_s{seed}.json")
    if os.path.exists(path):
        return json.load(open(path))
    c = copy.deepcopy(cfg)
    for p, v in overrides.items():
        set_path(c, p, v)
    t0 = time.perf_counter()
    algo, ctrl, log = train_policy(c, os.path.join(out, "runs", f"{safe(name)}_s{seed}"), seed, updates,
                                   verbose=False, plots=False, eval_every=max(1, updates // 5))
    tr, ho = layouts(c)
    held = evaluate(c, ctrl, ho, n_eval)
    train = evaluate(c, ctrl, tr, n_eval)
    rec = {"name": name, "seed": seed, "updates": updates, "seconds": round(time.perf_counter() - t0, 1),
           "heldout": {k: held[k] for k in METRICS}, "train_found": train["found_frac"],
           "gap": train["found_frac"] - held["found_frac"],
           "curve": [(l["update"], l["eval_heldout_found"]) for l in log if "eval_heldout_found" in l],
           "final_entropy": log[-1]["entropy"], "final_kl": log[-1]["approx_kl"], "final_clip": log[-1]["clip_frac"]}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(rec, open(path, "w"), indent=1)
    return rec


def sd(v):
    return float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")


def cmd_sweep(a, cfg):
    x = cfg["experiments"]
    groups = set((a.groups or "base,lr,epochs,minibatch,gamma,pool,reward,model").split(","))
    seeds = a.seeds or x["seeds"]
    updates, n_eval = a.updates or x["updates"], x["eval_episodes"]
    vs = variants(cfg, groups)
    recs = {n: [] for n in vs}
    for n, (g, ov) in vs.items():
        for s in seeds:
            print(f"[{n} | seed {s}] ...", flush=True)
            recs[n].append(run_variant(cfg, n, ov, s, updates, n_eval, a.out))
    rows = []
    hdr = f"{'variant':26s} {'held found':>14s} {'train found':>12s} {'gap':>7s} {'return':>8s} {'coll':>6s} {'dead':>5s} {'loops':>6s} {'entropy':>8s}"
    print(f"\n{len(seeds)} seeds, {updates} updates/variant; mean+-SD across seeds (descriptive)\n" + hdr)
    for n, rs in recs.items():
        f = [r["heldout"]["found_frac"] for r in rs]
        line = (f"{n:26s} {np.mean(f):6.3f}+-{sd(f):5.3f} {np.mean([r['train_found'] for r in rs]):12.3f} "
                f"{np.mean([r['gap'] for r in rs]):+7.3f} {np.mean([r['heldout']['ret'] for r in rs]):8.1f} "
                f"{np.mean([r['heldout']['collisions'] for r in rs]):6.1f} {np.mean([r['heldout']['dead'] for r in rs]):5.2f} "
                f"{np.mean([r['heldout']['loops'] for r in rs]):6.1f} {np.mean([r['final_entropy'] for r in rs]):8.3f}")
        print(line); rows.append(line)
    open(os.path.join(a.out, "sweep_table.txt"), "w").write(hdr + "\n" + "\n".join(rows) + "\n")
    names = list(recs)
    m = [np.mean([r["heldout"]["found_frac"] for r in recs[n]]) for n in names]
    e = [sd([r["heldout"]["found_frac"] for r in recs[n]]) for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].bar(names, m, yerr=np.nan_to_num(e)); ax[0].set_title("Held-out victims confirmed (mean +- SD over seeds)")
    ax[0].tick_params(axis="x", rotation=75)
    for n in names:                                                    # learning-speed curves
        cs = np.array([[v for _, v in r["curve"]] for r in recs[n]])
        ax[1].plot([u for u, _ in recs[n][0]["curve"]], cs.mean(0), label=n)
    ax[1].set_title("Held-out learning curves"); ax[1].set_xlabel("PPO update"); ax[1].legend(fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "fig_sweep.png"), dpi=130); plt.close()
    json.dump(recs, open(os.path.join(a.out, "sweep_results.json"), "w"), indent=1)


def cmd_forgetting(a, cfg):
    """Train on task A, then continue on a different task B (denser rubble). Catastrophic forgetting =
    drop of held-out performance on A. Replay of A episodes during phase B is the mitigation tested."""
    import torch
    from train import build, train_phase
    x = cfg["experiments"]; fx = x["forgetting"]
    seeds = a.seeds or x["seeds"]
    cfg_b = copy.deepcopy(cfg); cfg_b["env"]["obstacle_density"] = fx["env_b_obstacle_density"]
    n = x["eval_episodes"]; os.makedirs(a.out, exist_ok=True)
    res = {"finetune_B_only": [], "finetune_B_with_replay_A": []}
    for seed in seeds:
        c = copy.deepcopy(cfg); c["seed"] = seed
        torch.manual_seed(seed); np.random.seed(seed)
        _, algo, ctrl, tr, ho = build(c, seed)
        rng = np.random.default_rng(seed)
        print(f"[seed {seed}] phase A: {fx['phase_a_updates']} updates on task A", flush=True)
        train_phase(c, algo, ctrl, [(c, tr, 1.0)], fx["phase_a_updates"], rng, {}, verbose=False)
        ev = lambda ct, cc, c_env, lay: evaluate(c_env, ct, lay, n)["found_frac"]
        a0, b0 = ev(ctrl, c, c, ho), ev(ctrl, c, cfg_b, ho)
        for cond, replay in (("finetune_B_only", 0.0), ("finetune_B_with_replay_A", fx["replay_fraction"])):
            alg = copy.deepcopy(algo)
            ct = SwarmController(alg, FuzzySafetyFilter(c, seed + 1), c, "mappo", seed)
            src = [(cfg_b, tr, 1.0 - replay)] + ([(c, tr, replay)] if replay > 0 else [])
            train_phase(c, alg, ct, src, fx["phase_b_updates"], np.random.default_rng(seed + 777), {},
                        verbose=False, start=fx["phase_a_updates"])
            a1, b1 = ev(ct, c, c, ho), ev(ct, c, cfg_b, ho)
            res[cond].append(dict(seed=seed, A_before=a0, B_before=b0, A_after=a1, B_after=b1,
                                  forgetting=a0 - a1, gain_B=b1 - b0))
    print(f"\ntask A: obstacle density {cfg['env']['obstacle_density']}, task B: {fx['env_b_obstacle_density']}; "
          f"held-out layouts; victims confirmed")
    for cond, rs in res.items():
        g = lambda k: (np.mean([r[k] for r in rs]), sd([r[k] for r in rs]))
        print(f"{cond:26s} A {g('A_before')[0]:.3f}->{g('A_after')[0]:.3f} | B {g('B_before')[0]:.3f}->{g('B_after')[0]:.3f}"
              f" | forgetting(A) {g('forgetting')[0]:+.3f}+-{g('forgetting')[1]:.3f}")
    json.dump(res, open(os.path.join(a.out, "forgetting_results.json"), "w"), indent=1)


def cmd_attention(a, cfg):
    """Cost of over-trust: mission outcome and automation-bias rate vs simulated supervisor scrutiny."""
    from mappo import MAPPO
    from xai import ApprovalGate, ScriptedSupervisor
    tr, ho = layouts(cfg)
    e0 = SearchEnv(cfg, tr[0])
    algo = MAPPO(e0.obs_dim, e0.state_dim, e0.n_actions, cfg["mappo"])
    algo.load(os.path.join(a.run_dir, "mappo.pt"))
    n, seed = cfg["experiments"]["eval_episodes"], cfg["seed"] if a.seed is None else a.seed
    mk = lambda: SwarmController(algo, FuzzySafetyFilter(cfg, seed + 1), cfg, "mappo")
    rows = {"no_gate": {**{k: v for k, v in evaluate(cfg, mk(), ho, n).items() if k in METRICS},
                        "reviews_per_ep": 0.0, "bias_rate": float("nan")}}
    for att in cfg["experiments"]["attention"]:
        gate = ApprovalGate(cfg, ScriptedSupervisor(att, 3), None, 4)
        r = evaluate(cfg, mk(), ho, n, gate=gate)
        rows[f"attention={att}"] = {**{k: r[k] for k in METRICS}, "reviews_per_ep": gate.stats["reviews"] / n,
                                    "bias_rate": gate.automation_bias_rate(),
                                    "rejected_per_ep": gate.stats["rejected"] / n}
    print(f"{'condition':16s} {'found':>6s} {'return':>8s} {'dead':>5s} {'coll':>6s} {'reviews/ep':>10s} {'decoys approved':>16s}")
    for k, r in rows.items():
        print(f"{k:16s} {r['found_frac']:6.3f} {r['ret']:8.1f} {r['dead']:5.2f} {r['collisions']:6.1f} "
              f"{r['reviews_per_ep']:10.1f} {r['bias_rate']:16.2f}")
    print("NOTE: supervisor is SIMULATED (ScriptedSupervisor); decoy approval ~ 1 - attention by construction. "
          "The informative columns are the mission outcomes and the review workload, not the bias rate itself.")
    json.dump(rows, open(os.path.join(a.run_dir, "attention_sweep.json"), "w"), indent=1)
    names = list(rows)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for x_, key, t in zip(ax, ("found_frac", "reviews_per_ep", "bias_rate"),
                          ("Victims confirmed", "Reviews per episode (operator workload)", "Decoy plans approved")):
        x_.bar(names, [rows[k][key] for k in names]); x_.set_title(t); x_.tick_params(axis="x", rotation=60)
    plt.tight_layout(); plt.savefig(os.path.join(a.run_dir, "fig_attention_sweep.png"), dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep", "forgetting", "attention"])
    ap.add_argument("--config", default="configs/agent.yaml")
    ap.add_argument("--out", default="runs/experiments")
    ap.add_argument("--run_dir", default="runs/seed0")
    ap.add_argument("--groups", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    {"sweep": cmd_sweep, "forgetting": cmd_forgetting, "attention": cmd_attention}[a.cmd](a, cfg)


if __name__ == "__main__":
    main()
