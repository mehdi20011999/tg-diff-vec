"""Baselines, privileged expert, and diffusion policies (Diff and TG-Diff)."""
import math
import time
import numpy as np
import torch
import torch.nn as nn

from .env import NSLOT, NFEAT, OBS_DIM, cost

TRUST_IDX, DEV_IDX = 9, 10  # feature positions inside a slot


def masked(slots):
    return [i for i, s in enumerate(slots) if s is not None]


# ---------------- classic policies ----------------
def greedy(env, v, task, slots, obs):
    c = masked(slots)
    return min(c, key=lambda i: cost(env, slots[i], task)), 0.0


def greedy_trust(env, v, task, slots, obs, use_dt=True, thr=0.5):
    c = [i for i in masked(slots) if slots[i]["node"] is None or slots[i]["node"].kind == "cloud"
         or (slots[i]["node"].s + slots[i]["node"].f < 3 and not use_dt) or env.trust(slots[i]["node"], use_dt) >= thr]
    return min(c or [0], key=lambda i: cost(env, slots[i], task)), 0.0


def expert(env, v, task, slots, obs):
    """Privileged expert ("learning by cheating"): true backlogs and ground-truth maliciousness."""
    c = [i for i in masked(slots) if slots[i]["node"] is None or not slots[i]["node"].mal]
    return min(c, key=lambda i: cost(env, slots[i], task, truth=True)), 0.0


# ---------------- diffusion model ----------------
def tembed(t, dim=16):
    half = dim // 2
    f = torch.exp(-math.log(1000.0) * torch.arange(half, dtype=torch.float32) / half)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], dim=1)


class Denoiser(nn.Module):
    def __init__(self, cond_dim=OBS_DIM, hid=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(NSLOT + 16 + cond_dim, hid), nn.Mish(), nn.Linear(hid, hid), nn.Mish(),
                                 nn.Linear(hid, hid), nn.Mish(), nn.Linear(hid, NSLOT))

    def forward(self, x, t, c):
        return self.net(torch.cat([x, tembed(t), c], dim=1))


class DiffusionPolicy:
    """DDPM training (T steps), DDIM sampling with K <= T steps. Action = argmax of the sampled slot vector.

    use_trust_input: trust and twin-deviation features are visible to the denoiser
    guidance: lambda * log(trust) is added to the predicted clean sample at every step (trust-guided sampling)
    """
    T = 20

    def __init__(self, use_trust_input=True, guidance=0.0, use_dt=True, steps=10, seed=0):
        torch.manual_seed(seed)
        self.use_trust_input, self.guidance, self.use_dt, self.steps = use_trust_input, guidance, use_dt, steps
        self.model = Denoiser()
        b = torch.linspace(1e-4, 0.2, self.T)
        self.abar = torch.cumprod(1 - b, 0)
        self.gen = torch.Generator().manual_seed(seed)

    def cond(self, obs):
        c = obs.copy()
        if not self.use_trust_input:
            for i in range(NSLOT):
                c[..., i * NFEAT + TRUST_IDX] = 0.0
                c[..., i * NFEAT + DEV_IDX] = 0.0
        return c

    def fit(self, obs, act, iters=4000, bs=512, lr=1e-3, log=None):
        X = torch.tensor(self.cond(obs))
        mask = torch.tensor(obs[:, 0::NFEAT][:, :NSLOT] > 0)
        A = -torch.ones(len(act), NSLOT)
        A[torch.arange(len(act)), torch.tensor(act)] = 1.0
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        for it in range(iters):
            idx = torch.randint(0, len(X), (bs,))
            x0, c = A[idx], X[idx]
            t = torch.randint(0, self.T, (bs,))
            eps = torch.randn_like(x0)
            ab = self.abar[t][:, None]
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
            loss = ((self.model(xt, t, c) - eps) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if log and it % 1000 == 0:
                log(f"  iter {it} loss {loss.item():.4f}")
        return self

    @torch.no_grad()
    def sample(self, obs, trust_vec, mask):
        K = self.steps
        ts = torch.linspace(self.T - 1, 0, K).round().long()
        c = torch.tensor(self.cond(obs))[None]
        x = torch.randn(1, NSLOT, generator=self.gen)
        g = torch.tensor(np.log(np.clip(trust_vec, 1e-3, 1.0)), dtype=torch.float32)[None]
        for j, t in enumerate(ts):
            ab = self.abar[t]
            eps = self.model(x, t[None], c)
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1.5, 1.5)
            if self.guidance:
                x0 = x0 + self.guidance * g
            if j + 1 < K:
                abn = self.abar[ts[j + 1]]
                x = abn.sqrt() * x0 + (1 - abn).sqrt() * eps  # deterministic DDIM step
            else:
                x = x0
        x = x[0].numpy()
        x[~mask] = -1e9
        return int(np.argmax(x))

    def __call__(self, env, v, task, slots, obs):
        mask = np.array([s is not None for s in slots])
        tv = np.ones(NSLOT)
        for i, s in enumerate(slots):
            if s is not None and s["node"] is not None and s["node"].kind != "cloud":
                tv[i] = env.trust(s["node"], self.use_dt)
        t0 = time.perf_counter()
        a = self.sample(obs, tv, mask)
        return a, time.perf_counter() - t0


class BCMLP:
    """Non-generative behaviour-cloning classifier with the same inputs (trust features visible)."""

    def __init__(self, seed=0):
        torch.manual_seed(seed)
        self.model = nn.Sequential(nn.Linear(OBS_DIM, 256), nn.Mish(), nn.Linear(256, 256), nn.Mish(), nn.Linear(256, NSLOT))

    def fit(self, obs, act, iters=4000, bs=512, lr=1e-3, log=None):
        X, Y = torch.tensor(obs), torch.tensor(act)
        M = torch.tensor(obs[:, 0::NFEAT][:, :NSLOT] > 0)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        for it in range(iters):
            idx = torch.randint(0, len(X), (bs,))
            logits = self.model(X[idx]).masked_fill(~M[idx], -1e9)
            loss = nn.functional.cross_entropy(logits, Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        return self

    @torch.no_grad()
    def __call__(self, env, v, task, slots, obs):
        t0 = time.perf_counter()
        z = self.model(torch.tensor(obs)[None])[0].numpy()
        z[[s is None for s in slots]] = -1e9
        return int(np.argmax(z)), time.perf_counter() - t0
