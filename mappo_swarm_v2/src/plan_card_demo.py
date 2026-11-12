"""Print and save real plan cards (the XAI screenshot source).
Usage: python src/plan_card_demo.py --config configs/agent.yaml --run_dir runs/seed0 --seed 0
Writes runs/seed0/plan_card.txt and plan_card.png (first non-decoy, vetoed plan if any).
evaluate.py --xai-demo only writes execution.log and audit.jsonl; it never prints a card."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from env import SearchEnv
from runtime import SwarmController, run_episode
from safety import FuzzySafetyFilter
from xai import ApprovalGate, ScriptedSupervisor, render_flow, render_text


class Capture(ScriptedSupervisor):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.plans = []
    def decide(self, plan):
        self.plans.append(plan); return super().decide(plan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/agent.yaml")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mode", default="mappo", choices=["mappo", "heuristic"])
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.seed is not None:
        cfg["seed"] = a.seed
    ho = list(range(1000, 1000 + cfg["train"]["heldout_layout_seeds"]))
    algo = None
    if a.mode == "mappo":
        from mappo import MAPPO
        e0 = SearchEnv(cfg, 0)
        algo = MAPPO(e0.obs_dim, e0.state_dim, e0.n_actions, cfg["mappo"])
        algo.load(os.path.join(a.run_dir, "mappo.pt"))
    ctrl = SwarmController(algo, FuzzySafetyFilter(cfg, cfg["seed"] + 1), cfg, a.mode)
    audit = os.path.join(a.run_dir, "plan_card_audit.jsonl")
    if os.path.exists(audit):
        os.remove(audit)
    sup = Capture(cfg["xai"]["supervisor_attention"], 3)
    gate = ApprovalGate(cfg, sup, audit, 4)              # same seeds as evaluate.py --xai-demo
    run_episode(SearchEnv(cfg, ho[0]), ctrl, 777, stochastic=False, gate=gate)
    decoys = {r["plan_id"] for r in map(json.loads, open(audit)) if r["is_challenge"]}
    real = [p for p in sup.plans if p["plan_id"] not in decoys]
    pick = next((p for p in real if p["filter"]["vetoed"]), real[0])
    text = render_text(pick) + "\n  " + render_flow(pick)
    print(text)
    open(os.path.join(a.run_dir, "plan_card.txt"), "w").write(text + "\n")
    fig = plt.figure(figsize=(14, 3.2)); fig.patch.set_facecolor("white")
    fig.text(0.01, 0.97, text, family="monospace", fontsize=7.5, va="top", wrap=True)
    fig.savefig(os.path.join(a.run_dir, "plan_card.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
