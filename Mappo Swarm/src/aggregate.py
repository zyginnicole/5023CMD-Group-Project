"""Aggregate per-seed results: python src/aggregate.py runs/seed0 runs/seed1 runs/seed2
Reports mean +- SD ACROSS SEEDS (sample SD, ddof=1) of each condition's mean metric."""
import json
import sys

import numpy as np

runs = [json.load(open(p.rstrip("/") + "/results.json")) for p in sys.argv[1:]]
print(f"{len(runs)} seeds, sample SD (descriptive only; no significance claimed)")
sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
for cond in runs[0]:
    row = []
    for m in ("found_frac", "ret", "collisions", "dead", "home", "vetoes", "loops",
              "return_at_base", "return_moving"):
        v = [r[cond][m] for r in runs]
        row.append(f"{m}={np.mean(v):.2f}+-{sd(v):.2f}")
    print(f"{cond:28s} " + "  ".join(row))

gaps = [r["mappo_train_layouts"]["found_frac"] - r["mappo_heldout_layouts"]["found_frac"] for r in runs]
print(f"generalisation gap (train - held-out found_frac) = {np.mean(gaps):+.3f}+-{sd(gaps):.3f}")
import os
tm = [json.load(open(p.rstrip("/") + "/timing.json")) for p in sys.argv[1:]
      if os.path.exists(p.rstrip("/") + "/timing.json")]
for key in ("train_seconds", "eval_seconds"):
    v = [t[key] for t in tm if key in t]
    if v:
        print(f"{key}: " + ", ".join(f"{x:.0f}" for x in v)
              + f"  (mean {np.mean(v):.0f} s = {np.mean(v) / 60:.1f} min)")
if tm:
    print("hardware:", tm[0].get("machine"), "| python", tm[0].get("python"), "| torch", tm[0].get("torch"),
          "| cuda", tm[0].get("cuda"))
