"""Offline dataset from the privileged expert, with optional label poisoning."""
import numpy as np
from .env import VECEnv, NSLOT
from .policies import expert


def collect(episodes=30, seconds=60.0, seed=100, log=None):
    O, A, MAL = [], [], []
    rng = np.random.default_rng(seed)
    for e in range(episodes):
        env = VECEnv(nveh=int(rng.integers(20, 81)), rate_=float(rng.uniform(0.3, 1.0)),
                     mal_frac=float(rng.uniform(0.0, 0.5)), black_hole=bool(rng.random() < 0.5),
                     false_reports=True, seed=seed + e)

        def rec(env, v, task, slots, obs, a):
            O.append(obs)
            A.append(a)
            MAL.append([s is not None and s["node"] is not None and s["node"].mal for s in slots])

        env.run(expert, seconds, record=rec)
        if log:
            log(f"  episode {e+1}/{episodes}: {len(A)} samples")
    return np.array(O, dtype=np.float32), np.array(A), np.array(MAL, dtype=bool)


def poison(A, MAL, rate, seed=0):
    """Data poisoning: a fraction `rate` of the samples that offer a malicious slot are relabelled to it."""
    rng = np.random.default_rng(seed)
    A = A.copy()
    cand = np.where(MAL.any(1))[0]
    hit = cand[rng.random(len(cand)) < rate]
    for i in hit:
        A[i] = rng.choice(np.where(MAL[i])[0])
    return A, len(hit)
