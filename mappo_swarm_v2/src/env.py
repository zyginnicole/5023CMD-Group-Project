"""Search-and-rescue grid world: a small Dec-POMDP stand-in for the Task 1 swarm.

Each drone sees only a local k x k patch of the shared Bayesian belief, its own
battery / position uncertainty, and neighbour link quality (Table 1 in the brief).
"""
from collections import deque

import numpy as np

ACTIONS = ["advance", "hold", "reroute", "verify", "return"]
ADV, HOLD, REROUTE, VERIFY, RETURN = range(5)
MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # N, S, W, E


def entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


class SearchEnv:
    def __init__(self, cfg, layout_seed, noise_scale=1.0):
        self.c, self.r = cfg["env"], cfg["reward"]
        self.H = self.W = self.c["size"]
        self.N, self.k = self.c["n_drones"], self.c["patch"]
        self.pad, self.noise = self.k // 2, noise_scale
        self.n_actions = len(ACTIONS)
        self.base = (0, 0)
        # sensor noise grows with noise_scale (used for reality-gap tests)
        self.pd = max(0.55, 1 - (1 - self.c["pd"]) * noise_scale)
        self.pf = min(0.40, self.c["pf"] * noise_scale)
        self._build_layout(layout_seed)
        self.obs_dim = 3 * self.k * self.k + 9
        self.state_dim = self.N * self.obs_dim + 2 * self.H * self.W

    # ---------- layout ----------
    def _build_layout(self, seed):
        rng = np.random.default_rng(seed)
        while True:                                        # redraw until enough of the grid is reachable
            self.obst = rng.random((self.H, self.W)) < self.c["obstacle_density"]
            self.obst[self.base] = False
            seen, q = {self.base}, deque([self.base])      # BFS from base
            while q:
                y, x = q.popleft()
                for dy, dx in MOVES:
                    n = (int(y + dy), int(x + dx))
                    if self._valid(*n) and n not in seen:
                        seen.add(n); q.append(n)
            if len(seen) - 1 >= max(self.c["n_victims"], 0.6 * self.H * self.W):
                break
        cells = sorted(seen - {self.base})
        pick = rng.choice(len(cells), self.c["n_victims"], replace=False)
        self.victim = np.zeros((self.H, self.W), bool)
        for j in pick:
            self.victim[cells[j]] = True
        self.n_victims = int(self.victim.sum())

    def _valid(self, y, x):
        return 0 <= y < self.H and 0 <= x < self.W and not self.obst[y, x]

    # ---------- helpers used by filter / heuristic / XAI ----------
    def at_base(self, i):
        return tuple(self.pos[i]) == self.base

    def dist_base(self, i):
        return abs(self.pos[i][0] - self.base[0]) + abs(self.pos[i][1] - self.base[1])

    def battery_margin(self, i):
        need = self.dist_base(i) * self.c["cost_move"] * 1.2
        return (self.battery[i] - need) / self.c["battery_max"]

    def p_here(self, i):
        return float(self.belief[self.pos[i][0], self.pos[i][1]])

    def at_goal(self, i):
        return tuple(self.pos[i]) == tuple(self.goal[i])

    def _toward(self, src, dst):
        """Move indices that reduce Manhattan distance, larger axis first."""
        dy, dx = dst[0] - src[0], dst[1] - src[1]
        opts = []
        if dy: opts.append((abs(dy), 1 if dy > 0 else 0))
        if dx: opts.append((abs(dx), 3 if dx > 0 else 2))
        return [m for _, m in sorted(opts, reverse=True)]

    def blocked_ahead(self, i):
        cand = self._toward(self.pos[i], self.goal[i])[:1]
        if not cand:
            return False
        ny, nx = self.pos[i] + MOVES[cand[0]]
        return not self._valid(ny, nx)

    # ---------- allocator stub (stands in for the Task 1 LLM allocator) ----------
    def assign_goals(self):
        var = self.belief * (1 - self.belief)
        var[self.resolved | self.obst] = 0.0
        yy, xx = np.mgrid[0:self.H, 0:self.W]
        taken = set()
        for i in range(self.N):
            y, x = self.pos[i]
            score = var - 0.02 * (abs(yy - y) + abs(xx - x)) / (self.H + self.W)
            score[self.obst] = -9
            for t in taken:
                score[t] = -9
            g = np.unravel_index(np.argmax(score), score.shape)
            self.goal[i] = g
            taken.add(g)

    # ---------- episode ----------
    def reset(self, episode_seed):
        self.rng = np.random.default_rng(episode_seed)
        self.pos = np.tile(self.base, (self.N, 1)).astype(int)
        self.battery = np.full(self.N, self.c["battery_max"])
        self.sigma = np.zeros(self.N)
        self.dead = np.zeros(self.N, bool)
        self.belief = np.full((self.H, self.W), self.c["prior"])
        self.belief[self.obst] = 0.0
        self.belief[self.base] = 0.0
        self.resolved = np.zeros((self.H, self.W), bool)
        self.goal = np.zeros((self.N, 2), int)
        self.found, self.t = 0, 0
        self.assign_goals()
        return self.observe()

    def _scan(self, i, radius, pd, pf):
        """Bayesian update of cells around drone i; returns entropy reduction."""
        y, x = self.pos[i]
        ys = np.arange(max(0, y - radius), min(self.H, y + radius + 1))
        xs = np.arange(max(0, x - radius), min(self.W, x + radius + 1))
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        yy, xx = yy.ravel(), xx.ravel()
        keep = ~(self.obst[yy, xx] | self.resolved[yy, xx])
        yy, xx = yy[keep], xx[keep]
        if len(yy) == 0:
            return 0.0
        z = self.rng.random(len(yy)) < np.where(self.victim[yy, xx], pd, pf)
        p = self.belief[yy, xx]
        l1, l0 = np.where(z, pd, 1 - pd), np.where(z, pf, 1 - pf)
        pn = p * l1 / (p * l1 + (1 - p) * l0)
        self.belief[yy, xx] = pn
        return float((entropy(p) - entropy(pn)).sum())

    def _try_move(self, i, cands):
        for m in cands:
            ny, nx = self.pos[i] + MOVES[m]
            if self._valid(ny, nx):
                self.pos[i] = (ny, nx)
                return True
        return False

    def _verify(self, i, info):
        c, r = self.c, self.r
        y, x = self.pos[i]
        if self.resolved[y, x]:
            return r["waste"]
        p, rew = self.belief[y, x], 0.0
        if p < c["waste_p"]:
            rew += r["waste"]
        truth = self.victim[y, x]
        pd, pf = c["pd_verify"], c["pf_verify"]
        z = self.rng.random() < (pd if truth else pf)
        l1, l0 = (pd if z else 1 - pd), (pf if z else 1 - pf)
        pn = p * l1 / (p * l1 + (1 - p) * l0)
        if z and pn >= c["confirm_p"]:
            self.resolved[y, x], self.belief[y, x] = True, 0.0
            if truth:
                info["found"] += 1; rew += r["find"]
            else:
                info["false_pos"] += 1; rew += r["false_pos"]
        else:
            self.belief[y, x] = pn
        return rew

    def step(self, actions, vetoed=None):
        c, r = self.c, self.r
        rew, info = 0.0, dict(found=0, collisions=0, false_pos=0, dead=0)
        for i in range(self.N):
            if self.dead[i]:
                continue
            a, cost, moved = int(actions[i]), c["cost_move"], False
            y, x = self.pos[i]
            primary = self._toward((y, x), self.goal[i])[:1]
            if a == ADV and primary:
                moved = self._try_move(i, primary)
                if not moved:
                    info["collisions"] += 1; rew += r["collision"]
            elif a == REROUTE:
                cands = [m for m in range(4) if not (primary and m == primary[0])
                         and self._valid(y + MOVES[m][0], x + MOVES[m][1])]
                gy, gx = self.goal[i]
                cands.sort(key=lambda m: abs(y + MOVES[m][0] - gy)
                           + abs(x + MOVES[m][1] - gx) + 0.01 * self.rng.random())
                moved = self._try_move(i, cands[:1])
            elif a == RETURN and not self.at_base(i):
                moved = self._try_move(i, self._toward((y, x), self.base))
                if not moved:
                    info["collisions"] += 1; rew += r["collision"]
            elif a == VERIFY:
                cost = c["cost_verify"]; rew += self._verify(i, info)
            elif a in (HOLD, RETURN, ADV):
                cost = c["cost_hold"]; self.sigma[i] *= 0.8
            if moved:
                self.sigma[i] = min(1.0, self.sigma[i] + c["sigma_growth"] * self.noise)
            self.battery[i] -= cost
            if self.at_base(i):
                self.sigma[i] = 0.0
                if a in (HOLD, RETURN):
                    self.battery[i] = min(c["battery_max"], self.battery[i] + c["recharge"])
            elif self.battery[i] <= 0:
                self.dead[i] = True; info["dead"] += 1; rew += r["dead"]
        alive = [i for i in range(self.N) if not self.dead[i]]
        for i in alive:                                    # passive Bayesian sensing
            rew += r["info"] * self._scan(i, c["sensor_radius"], self.pd, self.pf)
        rew += r["step"] * len(alive)
        cells = [tuple(self.pos[i]) for i in alive if not self.at_base(i)]
        rew += r["overlap"] * (len(cells) - len(set(cells)))
        if vetoed is not None:
            rew += r["veto"] * sum(bool(v) for v in vetoed)
        self.found += info["found"]
        self.t += 1
        done = (self.t >= c["max_steps"] or not alive or
                (self.found == self.n_victims and all(self.at_base(i) for i in alive)))
        if done:
            rew += r["return"] * sum(self.at_base(i) for i in alive)
        elif self.t % c["goal_refresh"] == 0 or any(self.at_goal(i) for i in alive):
            self.assign_goals()
        return self.observe(), rew, done, info

    # ---------- observations ----------
    def observe(self):
        k, p = self.k, self.pad
        bp = np.pad(self.belief, p)
        op = np.pad(self.obst.astype(float), p, constant_values=1.0)
        out = []
        for i in range(self.N):
            y, x = self.pos[i]
            mean = bp[y:y + k, x:x + k]
            patch = np.concatenate([mean.ravel(), (mean * (1 - mean)).ravel(),
                                    op[y:y + k, x:x + k].ravel()])
            others = [j for j in range(self.N) if j != i and not self.dead[j]]
            if others:
                j = min(others, key=lambda j: abs(self.pos[j] - self.pos[i]).sum())
                d = self.pos[j] - self.pos[i]
                link = max(0.0, 1 - abs(d).sum() / self.c["comm_radius"])
                nb = [d[0] / self.H, d[1] / self.W, link]
            else:
                nb = [0.0, 0.0, 0.0]
            sc = [self.battery[i] / self.c["battery_max"], self.sigma[i],
                  (self.goal[i][0] - y) / self.H, (self.goal[i][1] - x) / self.W,
                  (self.base[0] - y) / self.H, (self.base[1] - x) / self.W] + nb
            out.append(np.concatenate([patch, sc]))
        return np.asarray(out, np.float32)

    def global_state(self):
        """Centralised-critic input (training only): all observations + privileged
        ground truth (unresolved victims, obstacles). Never used at execution time."""
        privileged = np.concatenate([(self.victim & ~self.resolved).ravel(),
                                     self.obst.ravel()]).astype(np.float32)
        return np.concatenate([self.observe().ravel(), privileged])
