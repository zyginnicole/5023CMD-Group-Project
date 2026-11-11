"""Aggregate per-seed results: python src/aggregate.py runs/seed0 runs/seed1 runs/seed2
Reports mean +- SD ACROSS SEEDS of each condition's mean metric (n = number of seeds)."""
import json
import sys

import numpy as np

runs = [json.load(open(p.rstrip("/") + "/results.json")) for p in sys.argv[1:]]
print(f"{len(runs)} seeds (descriptive only; no significance claimed)")
for cond in runs[0]:
    row = []
    for m in ("found_frac", "ret", "collisions", "dead", "loops", "return_at_base"):
        v = [r[cond][m] for r in runs]
        row.append(f"{m}={np.mean(v):.2f}+-{np.std(v):.2f}")
    print(f"{cond:28s} " + "  ".join(row))
