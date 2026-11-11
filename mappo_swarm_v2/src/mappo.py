"""MAPPO: parameter-shared actor, centralised critic (CTDE), PPO clipped objective, GAE."""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


class MAPPO:
    def __init__(self, obs_dim, state_dim, n_actions, cfg):
        self.c, h = cfg, cfg["hidden"]
        self.actor = mlp(obs_dim, h, n_actions)      # ONE actor shared by every drone
        self.critic = mlp(state_dim, h, 1)           # sees global state; training only
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
    def value(self, state):
        return float(self.critic(torch.as_tensor(state)).item())

    def gae(self, rew, val, done):
        """rew/val/done: [E, T]. Episodes end at done (horizon treated as terminal)."""
        g, lam = self.c["gamma"], self.c["gae_lambda"]
        adv = np.zeros_like(rew)
        last = np.zeros(rew.shape[0])
        for t in reversed(range(rew.shape[1])):
            nxt = val[:, t + 1] if t + 1 < rew.shape[1] else 0.0
            nt = 1.0 - done[:, t]
            delta = rew[:, t] + g * nxt * nt - val[:, t]
            last = delta + g * lam * nt * last
            adv[:, t] = last
        return adv, adv + val

    def update(self, b):
        """b: dict of arrays shaped [E, T, ...] with b['mask'] marking valid steps."""
        c = self.c
        adv, ret = self.gae(b["rew"], b["val"], b["done"])
        m = b["mask"] > 0
        T = lambda x: torch.as_tensor(np.ascontiguousarray(x[m]), dtype=torch.float32)
        obs, state, logp0 = T(b["obs"]), T(b["state"]), T(b["logp"])
        act = torch.as_tensor(b["act"][m], dtype=torch.long)
        advt, rett, val0 = T(adv), T(ret), T(b["val"])
        advt = (advt - advt.mean()) / (advt.std() + 1e-8)
        n, stats = len(advt), []
        for _ in range(c["ppo_epochs"]):
            perm = torch.randperm(n)
            for s in range(0, n, c["minibatch_size"]):
                ix = perm[s:s + c["minibatch_size"]]
                dist = Categorical(logits=self.actor(obs[ix]))        # [B, N, A]
                lp = dist.log_prob(act[ix])
                ratio = torch.exp(lp - logp0[ix])
                a = advt[ix][:, None]
                surr = torch.min(ratio * a, torch.clamp(ratio, 1 - c["clip_eps"], 1 + c["clip_eps"]) * a)
                ent = dist.entropy().mean()
                v = self.critic(state[ix]).squeeze(-1)
                if c.get("value_clip", True):
                    vc = val0[ix] + torch.clamp(v - val0[ix], -c["clip_eps"], c["clip_eps"])
                    vloss = torch.max((v - rett[ix]) ** 2, (vc - rett[ix]) ** 2).mean()
                else:
                    vloss = ((v - rett[ix]) ** 2).mean()
                loss = -surr.mean() - c["entropy_coef"] * ent + c["value_coef"] * vloss
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                         c["max_grad_norm"])
                self.opt.step()
                stats.append([(-surr.mean()).item(), vloss.item(), ent.item(),
                              (logp0[ix] - lp).mean().item(),
                              ((ratio - 1).abs() > c["clip_eps"]).float().mean().item()])
        s = np.mean(stats, 0)
        return dict(policy_loss=s[0], value_loss=s[1], entropy=s[2], approx_kl=s[3], clip_frac=s[4])

    def save(self, path, dims):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "dims": dims}, path)

    def load(self, path):
        d = torch.load(path)
        self.actor.load_state_dict(d["actor"]); self.critic.load_state_dict(d["critic"])
