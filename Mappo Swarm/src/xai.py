"""Explainable-AI layer: multi-step plan preview, approval gate, audit log, bias metrics."""
import copy
import json
import time

import numpy as np

from env import ACTIONS, obs_groups
from runtime import heuristic_action
from safety import LoopGuard


def drivers(env, ctrl, i, action, top=3):
    """Which observation groups drove the learned policy's log-probability of `action` (gradient x input).
    Empty for rule-based controllers (no learned model to attribute)."""
    algo = getattr(ctrl, "algo", None)
    if algo is None:
        return []
    att = algo.attribution(env.observe()[i], obs_groups(env.k), action)
    return [(g, round(v, 3)) for g, v in sorted(att.items(), key=lambda kv: -kv[1])[:top]]


def build_plan(env, ctrl, dec, i, horizon, plan_id):
    """Plan card for drone i: proposal, confidence, alternatives, filter verdict, look-ahead."""
    sim, g = copy.deepcopy(env), LoopGuard(env.N, ctrl.limit)
    ctrl = copy.copy(ctrl); ctrl.filt = copy.deepcopy(ctrl.filt)   # look-ahead must not consume the live filter RNG
    obs, steps = sim.observe(), []
    for _ in range(horizon):                       # greedy look-ahead on a COPY of the world
        d = ctrl.decide(sim, obs, stochastic=False, guard=g)
        steps.append({"t": sim.t, "action": ACTIONS[d["exe"][i]],
                      "conf": round(float(d["probs"][i][d["prop"][i]]), 3),
                      "battery": round(float(sim.battery[i]), 1), "pos": sim.pos[i].tolist()})
        obs, _, done, _ = sim.step(d["exe"])
        if done:
            break
    p, fr = dec["probs"][i], dec["filt"][i]
    top = np.argsort(-p)[:3]
    return {"plan_id": plan_id, "drone": i, "t": env.t, "pos": env.pos[i].tolist(),
            "proposed_action": ACTIONS[dec["prop"][i]], "confidence": round(float(p[dec["prop"][i]]), 3),
            "alternatives": [(ACTIONS[a], round(float(p[a]), 3)) for a in top],
            "filter": {"rule": fr.rule, "risk": round(fr.risk, 3), "vetoed": fr.vetoed,
                       "attenuated": fr.attenuated, "result_action": ACTIONS[fr.action]},
            "policy_entropy": round(float(-(p * np.log(p + 1e-12)).sum()), 3),
            "drivers": drivers(env, ctrl, i, dec["prop"][i]),
            "belief_here": round(env.p_here(i), 3), "battery": round(float(env.battery[i]), 1),
            "sigma": round(float(env.sigma[i]), 3), "reference_action": ACTIONS[heuristic_action(env, i)],
            "lookahead": steps}


def render_text(plan):
    f = plan["filter"]
    lines = [f"PLAN {plan['plan_id']} | drone {plan['drone']} | t={plan['t']} | pos={plan['pos']}",
             f"  Proposed: {plan['proposed_action']} (confidence {plan['confidence']})",
             "  Alternatives: " + ", ".join(f"{a} {p}" for a, p in plan["alternatives"]),
             "  Key drivers (gradient x input, first-order): "
             + (", ".join(f"{g} {v * 100:.0f}%" for g, v in plan["drivers"]) or "n/a (rule-based controller)")
             + f" | policy entropy {plan['policy_entropy']}",
             f"  Context: battery={plan['battery']} sigma={plan['sigma']} P(victim here)={plan['belief_here']}",
             f"  Safety filter: rule={f['rule']} risk={f['risk']} -> {f['result_action']}"
             + (" [VETO]" if f["vetoed"] else " [ATTENUATED]" if f["attenuated"] else ""),
             "  Look-ahead (greedy, run on a copy of the simulator, so optimistic): " + " -> ".join(f"{s['action']}({s['conf']})" for s in plan["lookahead"])]
    return "\n".join(lines)


def render_flow(plan):
    f = plan["filter"]
    return (f"[Observation] -> [Policy: {plan['proposed_action']} @ {plan['confidence']}] -> "
            f"[Fuzzy filter: risk {f['risk']}] -> [Human review] -> [Execute: {f['result_action']}]")


class ScriptedSupervisor:
    """Simulated supervisor. With prob. `attention` it scrutinises the plan; otherwise it
    rubber-stamps it (automation bias). Results are SIMULATED, not human-study data."""

    def __init__(self, attention, seed=0):
        self.attention, self.rng = attention, np.random.default_rng(seed)

    def decide(self, plan):
        if self.rng.random() >= self.attention:
            return True, "rubber_stamp"
        ok = (plan["proposed_action"] == plan["reference_action"]
              or (plan["confidence"] >= 0.5 and plan["filter"]["risk"] < 0.4))
        return ok, "scrutinised"


class CLISupervisor:
    def decide(self, plan):
        print(render_text(plan)); print("  " + render_flow(plan))
        ok = input("  Approve? [y/N]: ").strip().lower() == "y"
        return ok, input("  Reason (required): ").strip() or "none_given"


class ApprovalGate:
    def __init__(self, cfg, supervisor, audit_path, seed=0):
        self.c, self.sup, self.path = cfg["xai"], supervisor, audit_path
        self.rng, self.n = np.random.default_rng(seed), 0
        self.stats = dict(reviews=0, challenges=0, challenges_approved=0, rejected=0)

    def is_high_risk(self, dec, i):
        fr = dec["filt"][i]
        conf = dec["probs"][i][dec["prop"][i]]
        return fr.vetoed or fr.risk >= self.c["risk_threshold"] or conf < self.c["confidence_threshold"]

    def review(self, env, ctrl, dec, i, t, ep):
        self.n += 1
        plan = build_plan(env, ctrl, dec, i, self.c["horizon"], f"P{self.n:05d}")
        challenge = self.rng.random() < self.c["challenge_rate"]
        action = dec["exe"][i]
        if challenge:                                  # decoy: deliberately poor proposal
            action = int(np.argsort(dec["probs"][i])[0])
            plan.update(proposed_action=ACTIONS[action], confidence=round(float(dec["probs"][i][action]), 3),
                        drivers=drivers(env, ctrl, i, action))
        t0 = time.time()
        ok, mode = self.sup.decide(plan)
        self.stats["reviews"] += 1
        self.stats["challenges"] += challenge
        self.stats["challenges_approved"] += (challenge and ok)
        self.stats["rejected"] += (not ok)
        rec = {"plan_id": plan["plan_id"], "ep": ep, "t": t, "drone": i, "proposed": plan["proposed_action"],
               "confidence": plan["confidence"], "risk": plan["filter"]["risk"], "approved": bool(ok),
               "review_mode": mode, "is_challenge": bool(challenge),
               "latency_s": round(time.time() - t0, 4)}
        if self.path:                                   # path=None: statistics only (used by sweeps)
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return (int(action), "APPROVED") if ok else (1, "REJECTED")   # rejected -> HOLD (index 1)

    def automation_bias_rate(self):
        s = self.stats
        return s["challenges_approved"] / s["challenges"] if s["challenges"] else float("nan")
