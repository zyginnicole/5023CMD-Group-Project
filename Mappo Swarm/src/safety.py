"""Fuzzy safety filter + loop-trap guard (sits between the policy and the flight controller)."""
from collections import deque, namedtuple

import numpy as np

from env import ADV, HOLD, REROUTE, RETURN, VERIFY

FilterResult = namedtuple("FilterResult", "action risk rule vetoed attenuated")


def low(x, lo, hi):    # membership 1 at x<=lo, falling linearly to 0 at x>=hi
    return float(np.clip((hi - x) / (hi - lo), 0, 1))


def high(x, lo, hi):   # membership 0 at x<=lo, rising linearly to 1 at x>=hi
    return float(np.clip((x - lo) / (hi - lo), 0, 1))


class FuzzySafetyFilter:
    """Rule base (max-aggregation; the strongest rule decides the safe fallback):
      R1  IF battery_margin is LOW           AND action in {advance, reroute, verify} -> RETURN
      R2  IF position_uncertainty is HIGH    AND action in {advance, reroute}         -> HOLD
      R3  IF obstacle is AHEAD               AND action == advance                    -> REROUTE
    risk >= veto_threshold                  -> action is replaced (veto)
    attenuate <= risk < veto_threshold      -> replaced with probability (risk-a)/(v-a)
    """

    def __init__(self, cfg, seed=0):
        self.c, self.rng = cfg["safety"], np.random.default_rng(seed)

    def apply(self, env, i, action):
        c, rules = self.c, []
        if action in (ADV, REROUTE, VERIFY):
            rules.append((low(env.battery_margin(i), c["margin_lo"], c["margin_hi"]),
                          "R1_low_battery", RETURN))
        if action in (ADV, REROUTE):
            rules.append((high(env.sigma[i], c["sigma_lo"], c["sigma_hi"]),
                          "R2_high_position_uncertainty", HOLD))
        if action == ADV and env.blocked_ahead(i):
            rules.append((1.0, "R3_obstacle_ahead", REROUTE))
        if not rules:
            return FilterResult(action, 0.0, "none", False, False)
        risk, rule, safe = max(rules, key=lambda r: r[0])
        if risk <= 0.0:
            return FilterResult(action, 0.0, "none", False, False)   # no rule active
        a, v = c["attenuate_threshold"], c["veto_threshold"]
        if risk >= v:
            return FilterResult(safe, risk, rule, True, False)
        if risk >= a and self.rng.random() < (risk - a) / (v - a):
            return FilterResult(safe, risk, rule, False, True)
        return FilterResult(action, risk, rule, False, False)


class LoopGuard:
    """Flags (a) the same cell+action repeated `limit` times, (b) a 2-cell oscillation.
    With intervene=False it is a passive detector: it still counts loops (so the
    no-guard ablation is measured on the same footing) but never alters actions."""

    def __init__(self, n, limit, intervene=True):
        self.limit, self.intervene = limit, intervene
        self.h = [deque(maxlen=limit) for _ in range(n)]

    def update(self, i, pos, action, at_base):
        self.h[i].append((tuple(pos), int(action)))
        h = list(self.h[i])
        if at_base or len(h) < self.limit:
            return False
        if all(x == h[0] for x in h):
            return True
        p = [q for q, _ in h]
        return p[-1] == p[-3] and p[-2] == p[-4] and p[-1] != p[-2]
