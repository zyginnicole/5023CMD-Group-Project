"""MAPPO (default) / IPPO (ablation): parameter-shared actor, PPO clipped objective, GAE.

critic: central -> MAPPO. One critic sees the global state (all observations + privileged ground
                 truth, TRAINING ONLY) and outputs one team value.
critic: local   -> IPPO ablation. Each drone's critic sees only its own observation.
Dead drones are masked out of every loss term: their proposals are not real decisions.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


class MAPPO:
    def __init__(self, obs_dim, state_dim, n_actions, cfg):
        self.c, h = cfg, cfg["hidden"]
        self.local = cfg.get("critic", "central") == "local"
        self.actor = mlp(obs_dim, h, n_actions)                       # ONE actor shared by every drone
        self.critic = mlp(obs_dim if self.local else state_dim, h, 1)  # training only
        lr = cfg["learning_rate"]
        self.opt = torch.optim.Adam(
            [{"params": self.actor.parameters(), "lr": lr},
             {"params": self.critic.parameters(), "lr": lr * cfg["critic_lr_scale"]}], eps=1e-5)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        logits = self.actor(torch.as_tensor(obs))
        dist = Categorical(logits=logits)
        a = logits.argmax(-1) if deterministic else dist.sample()
        return a.numpy(), dist.log_prob(a).numpy(), dist.probs.numpy()

    @torch.no_grad()
    def value(self, state, obs=None):
        """MAPPO: scalar team value from the global state. IPPO: one value per drone from its own obs."""
        if self.local:
            return self.critic(torch.as_tensor(obs)).squeeze(-1).numpy()
        return float(self.critic(torch.as_tensor(state)).item())

    def attribution(self, obs_i, groups, action=None):
        """Gradient x input saliency of log pi(action | obs), summed per observation group and
        normalised to shares. First-order, local explanation: NOT a causal statement."""
        x = torch.as_tensor(np.asarray(obs_i, np.float32)).clone().requires_grad_(True)
        logp = torch.log_softmax(self.actor(x), -1)
        a = int(logp.argmax()) if action is None else int(action)
        self.actor.zero_grad()
        logp[a].backward()
        s = (x.grad * x.detach()).abs().numpy()
        tot = float(s.sum()) + 1e-12
        return {g: float(s[idx].sum() / tot) for g, idx in groups.items()}

    def gae(self, rew, val, done):
        """rew/done: [E, T]; val: [E, T] (MAPPO) or [E, T, N] (IPPO; team reward is broadcast).
        Episodes end at done (horizon truncation is treated as terminal)."""
        g, lam = self.c["gamma"], self.c["gae_lambda"]
        extra = val.ndim - rew.ndim
        rew = np.broadcast_to(rew.reshape(rew.shape + (1,) * extra), val.shape)
        done = np.broadcast_to(done.reshape(done.shape + (1,) * extra), val.shape)
        adv = np.zeros(val.shape, np.float64)
        last = np.zeros_like(val[:, 0], dtype=np.float64)
        for t in reversed(range(val.shape[1])):
            nxt = val[:, t + 1] if t + 1 < val.shape[1] else 0.0
            nt = 1.0 - done[:, t]
            delta = rew[:, t] + g * nxt * nt - val[:, t]
            last = delta + g * lam * nt * last
            adv[:, t] = last
        return adv, adv + val

    def update(self, b):
        """b: dict of arrays shaped [E, T, ...]; b['mask'] marks valid steps, b['alive'] [E,T,N]
        marks drones that were alive when the action was taken."""
        c = self.c
        adv, ret = self.gae(b["rew"], b["val"], b["done"])
        m = b["mask"] > 0
        T = lambda x: torch.as_tensor(np.ascontiguousarray(x[m]), dtype=torch.float32)
        obs, state, logp0, alive = T(b["obs"]), T(b["state"]), T(b["logp"]), T(b["alive"])
        act = torch.as_tensor(b["act"][m], dtype=torch.long)
        advt, rett, val0 = T(adv), T(ret), T(b["val"])
        sel = (alive > 0) if self.local else torch.ones_like(advt, dtype=torch.bool)
        advt = (advt - advt[sel].mean()) / (advt[sel].std() + 1e-8)
        wmean = lambda x, w: (x * w).sum() / w.sum().clamp(min=1.0)    # mean over alive drones only
        n, stats, eps = len(advt), [], c["clip_eps"]
        for _ in range(c["ppo_epochs"]):
            perm = torch.randperm(n)
            for s in range(0, n, c["minibatch_size"]):
                ix = perm[s:s + c["minibatch_size"]]
                al = alive[ix]                                          # [B, N]
                dist = Categorical(logits=self.actor(obs[ix]))          # [B, N, A]
                lp = dist.log_prob(act[ix])
                ratio = torch.exp(lp - logp0[ix])
                a = advt[ix] if self.local else advt[ix][:, None]
                surr = torch.min(ratio * a, torch.clamp(ratio, 1 - eps, 1 + eps) * a)
                ent = wmean(dist.entropy(), al)
                v = self.critic(obs[ix] if self.local else state[ix]).squeeze(-1)
                vw = al if self.local else torch.ones_like(v)
                if c.get("value_clip", True):
                    vc = val0[ix] + torch.clamp(v - val0[ix], -eps, eps)
                    vl = torch.max((v - rett[ix]) ** 2, (vc - rett[ix]) ** 2)
                else:
                    vl = (v - rett[ix]) ** 2
                vloss = wmean(vl, vw)
                pl = -wmean(surr, al)
                loss = pl - c["entropy_coef"] * ent + c["value_coef"] * vloss
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                         c["max_grad_norm"])
                self.opt.step()
                stats.append([pl.item(), vloss.item(), ent.item(),
                              wmean(logp0[ix] - lp, al).item(),
                              wmean(((ratio - 1).abs() > eps).float(), al).item()])
        s = np.mean(stats, 0)
        return dict(policy_loss=s[0], value_loss=s[1], entropy=s[2], approx_kl=s[3], clip_frac=s[4])

    def save(self, path, dims):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "dims": dims,
                    "critic_mode": "local" if self.local else "central"}, path)

    def load(self, path):
        d = torch.load(path, map_location="cpu")
        if d.get("critic_mode", "central") != ("local" if self.local else "central"):
            raise ValueError("checkpoint critic mode differs from cfg['mappo']['critic']")
        self.actor.load_state_dict(d["actor"]); self.critic.load_state_dict(d["critic"])
