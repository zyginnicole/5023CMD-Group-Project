"""Controller (policy -> fuzzy filter -> loop guard) and the episode runner."""
import json
import time
from collections import defaultdict

import numpy as np

from env import ACTIONS, ADV, HOLD, REROUTE, RETURN, VERIFY
from safety import FilterResult, LoopGuard


def heuristic_action(env, i):
    """Hand-written baseline policy (also used as the supervisor's reference judgement
    and as the deployment fallback if the learned policy fails)."""
    if env.battery_margin(i) < 0.1:
        return RETURN
    if env.p_here(i) > 0.3 and not env.resolved[env.pos[i][0], env.pos[i][1]]:
        return VERIFY
    return ADV


class SwarmController:
    def __init__(self, algo, filt, cfg, mode="mappo", seed=0):
        self.algo, self.filt, self.mode = algo, filt, mode
        self.limit = cfg["safety"]["repeat_limit"]
        self.rng = np.random.default_rng(seed)
        self.budget_ms = cfg.get("deployment", {}).get("latency_budget_ms")
        self.fallbacks, self.latency_violations = 0, 0      # deployment health counters

    def _heuristic(self, env, n, A):
        a = np.array([heuristic_action(env, i) for i in range(n)])
        pr = np.full((n, A), 0.1 / (A - 1)); pr[np.arange(n), a] = 0.9
        return a, np.log(pr[np.arange(n), a]), pr

    def propose(self, env, obs, stochastic):
        n, A = env.N, env.n_actions
        if self.mode == "mappo":
            t0 = time.perf_counter()
            try:
                out = self.algo.act(obs, deterministic=not stochastic)
            except Exception:                  # e.g. non-finite logits from a corrupt checkpoint
                if stochastic:
                    raise                      # never mask bugs during training
                self.fallbacks += 1
                return self._heuristic(env, n, A)
            if self.budget_ms is not None and (time.perf_counter() - t0) * 1e3 > self.budget_ms:
                self.latency_violations += 1
            return out
        if self.mode == "random":
            return (self.rng.integers(0, A, n), np.full(n, np.log(1 / A)), np.full((n, A), 1 / A))
        return self._heuristic(env, n, A)

    def decide(self, env, obs, stochastic=True, guard=None, use_filter=True):
        prop, logp, probs = self.propose(env, obs, stochastic)
        exe, frs, loops = prop.copy(), [], [False] * env.N
        for i in range(env.N):
            if env.dead[i]:
                frs.append(FilterResult(HOLD, 0.0, "dead", False, False)); exe[i] = HOLD
                continue
            fr = (self.filt.apply(env, i, int(prop[i])) if use_filter
                  else FilterResult(int(prop[i]), 0.0, "off", False, False))
            exe[i] = fr.action; frs.append(fr)
            if guard is not None and guard.update(i, env.pos[i], exe[i], env.at_base(i)):
                loops[i] = True; guard.h[i].clear()
                if guard.intervene:
                    exe[i] = REROUTE                                  # escalate out of the trap
        return dict(prop=prop, exe=exe, logp=logp, probs=probs, filt=frs, loops=loops)


def run_episode(env, ctrl, ep_seed, stochastic=True, collect=False, gate=None,
                logger=None, use_filter=True, use_guard=True):
    obs = env.reset(ep_seed)
    guard = LoopGuard(env.N, ctrl.limit, intervene=use_guard)   # always counts loops
    st = defaultdict(float)
    buf = defaultdict(list)
    acts = np.zeros(env.n_actions)
    for t in range(env.c["max_steps"]):
        state = env.global_state() if collect else None
        alive = (~env.dead).astype(np.float32) if collect else None
        dec = ctrl.decide(env, obs, stochastic, guard, use_filter)
        exe = dec["exe"].copy()
        human = ["N/A"] * env.N
        if gate is not None:
            for i in range(env.N):
                if not env.dead[i] and gate.is_high_risk(dec, i):
                    exe[i], human[i] = gate.review(env, ctrl, dec, i, t, ep_seed)
        for i in range(env.N):
            if env.dead[i]:
                continue                       # a dead drone's proposals are not real decisions
            acts[dec["prop"][i]] += 1
            if dec["prop"][i] == RETURN:
                st["return_at_base" if env.at_base(i) else "return_moving"] += 1
        vetoed = [fr.vetoed for fr in dec["filt"]]
        nxt, r, done, info = env.step(exe, vetoed)
        if logger:
            for i in range(env.N):
                logger.info(json.dumps({"ep": ep_seed, "t": t, "drone": i,
                                        "proposed": ACTIONS[dec["prop"][i]], "executed": ACTIONS[exe[i]],
                                        "conf": round(float(dec["probs"][i][dec["prop"][i]]), 3),
                                        "risk": round(dec["filt"][i].risk, 3), "rule": dec["filt"][i].rule,
                                        "human": human[i], "loop": dec["loops"][i],
                                        "battery": round(float(env.battery[i]), 1)}))
        if collect:
            buf["obs"].append(obs); buf["state"].append(state); buf["act"].append(dec["prop"])
            buf["logp"].append(dec["logp"]); buf["rew"].append(r); buf["done"].append(float(done))
            buf["val"].append(ctrl.algo.value(state, obs)); buf["alive"].append(alive)
        st["ret"] += r; st["loops"] += sum(dec["loops"])
        st["vetoes"] += sum(vetoed); st["attenuated"] += sum(fr.attenuated for fr in dec["filt"])
        st["false_pos"] += info["false_pos"]; st["collisions"] += info["collisions"]
        obs = nxt
        if done:
            break
    st["found_frac"] = env.found / env.n_victims
    st["dead"] = float(env.dead.sum())
    st["home"] = float(sum(env.at_base(i) for i in range(env.N) if not env.dead[i]))
    st["steps"] = env.t
    st["actions"] = acts
    return dict(st), buf
