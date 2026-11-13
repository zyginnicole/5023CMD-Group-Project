"""Run from the project root:  python -m unittest discover -s tests -v
Torch-free tests run anywhere; GAE is tested with torch stubbed out if torch is absent."""
import os
import sys
import unittest
from unittest import mock

import numpy as np
import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
CFG = yaml.safe_load(open(os.path.join(ROOT, "configs", "agent.yaml")))

from env import ADV, HOLD, REROUTE, RETURN, VERIFY, SearchEnv, obs_groups   # noqa: E402
from runtime import SwarmController, run_episode                            # noqa: E402
from safety import FuzzySafetyFilter, LoopGuard, high, low                  # noqa: E402
from xai import ApprovalGate, ScriptedSupervisor                            # noqa: E402


class ZeroRng:                       # forces every detection draw to succeed
    def random(self, n=None):
        return np.zeros(n) if n is not None else 0.0


class T(unittest.TestCase):
    def test_bayes_update_matches_hand_calculation(self):
        e = SearchEnv(CFG, 0); e.reset(0); e.rng = ZeroRng()
        e.pos[0] = (5, 5); e.belief[5, 5] = 0.10; e.resolved[:] = False; e.obst[5, 5] = False
        pd, pf = e.pd, e.pf
        e._scan(0, 0, pd, pf)                                    # one positive reading, own cell only
        self.assertAlmostEqual(e.belief[5, 5], 0.10 * pd / (0.10 * pd + 0.90 * pf), places=6)

    def test_membership_functions(self):
        self.assertEqual(low(0, 0, 1), 1.0); self.assertEqual(low(1, 0, 1), 0.0)
        self.assertAlmostEqual(high(0.5, 0, 1), 0.5); self.assertEqual(high(2, 0, 1), 1.0)

    def test_fuzzy_rule_r1_low_battery_vetoes_to_return(self):
        e = SearchEnv(CFG, 0); e.reset(0)
        e.pos[0] = (6, 6); e.battery[0] = 5.0                    # needs ~14.4 to get home -> margin < 0
        fr = FuzzySafetyFilter(CFG, 0).apply(e, 0, ADV)
        self.assertTrue(fr.vetoed); self.assertEqual(fr.action, RETURN); self.assertEqual(fr.rule, "R1_low_battery")

    def test_filter_never_touches_hold_or_return(self):
        e = SearchEnv(CFG, 0); e.reset(0); e.battery[0] = 0.5; e.pos[0] = (6, 6)
        f = FuzzySafetyFilter(CFG, 0)
        for a in (HOLD, RETURN):
            self.assertEqual(f.apply(e, 0, a).action, a)

    def test_loop_guard_flags_repeats_and_oscillation(self):
        g = LoopGuard(1, 6)
        hits = [g.update(0, (2, 2), HOLD, False) for _ in range(6)]
        self.assertTrue(hits[-1] and not any(hits[:-1]))
        g = LoopGuard(1, 6)
        seq = [(1, 1), (1, 2)] * 3
        self.assertTrue(any(g.update(0, p, REROUTE, False) for p in seq))

    def test_observation_layout_covers_every_dim_once(self):
        e = SearchEnv(CFG, 0); idx = sorted(i for v in obs_groups(e.k).values() for i in v)
        self.assertEqual(idx, list(range(e.obs_dim))); self.assertEqual(e.reset(0).shape, (e.N, e.obs_dim))

    def test_episode_is_deterministic_given_seeds(self):
        def go():
            ctrl = SwarmController(None, FuzzySafetyFilter(CFG, 1), CFG, "heuristic")
            st, _ = run_episode(SearchEnv(CFG, 1000), ctrl, 5, stochastic=False)
            return st["ret"], st["found_frac"], st["collisions"]
        self.assertEqual(go(), go())

    def test_automation_bias_measurement_extremes(self):
        def rate(att):
            ctrl = SwarmController(None, FuzzySafetyFilter(CFG, 1), CFG, "heuristic")
            g = ApprovalGate(CFG, ScriptedSupervisor(att, 3), None, 4)
            for ep in range(6):
                run_episode(SearchEnv(CFG, 1000 + ep), ctrl, ep, stochastic=False, gate=g)
            return g.stats, g.automation_bias_rate()
        s0, r0 = rate(0.0)                                       # never scrutinises -> approves every decoy
        if s0["challenges"]:
            self.assertEqual(r0, 1.0)
        s1, r1 = rate(1.0)
        if s1["challenges"]:
            self.assertLess(r1, 1.0)

    def test_gae_matches_hand_calculation(self):
        try:
            import torch                                          # noqa: F401
            from mappo import MAPPO
        except ImportError:
            with mock.patch.dict(sys.modules, {k: mock.MagicMock() for k in
                                               ("torch", "torch.nn", "torch.distributions")}):
                sys.modules.pop("mappo", None)
                from mappo import MAPPO
        a = MAPPO.__new__(MAPPO); a.c = dict(gamma=0.9, gae_lambda=0.8)
        rew, val, done = np.array([[1., 0., 2.]]), np.array([[.5, .4, .3]]), np.array([[0., 0., 1.]])
        adv, ret = a.gae(rew, val, done)
        d2, d1, d0 = 2 - .3, 0 + .9 * .3 - .4, 1 + .9 * .4 - .5
        a2 = d2; a1 = d1 + .72 * a2; a0 = d0 + .72 * a1
        np.testing.assert_allclose(adv[0], [a0, a1, a2], atol=1e-9)
        np.testing.assert_allclose(ret, adv + val)
        v3 = np.stack([val, val], -1)                             # IPPO shape: [E, T, N]
        adv3, _ = a.gae(rew, v3, done)
        np.testing.assert_allclose(adv3[0, :, 1], [a0, a1, a2], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
