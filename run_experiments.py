"""Preliminary experiments E1-E5 for the TG-Diff draft. Writes results/results.json."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from tgdiff.data import collect, poison
from tgdiff.env import VECEnv
from tgdiff.policies import BCMLP, DiffusionPolicy, expert, greedy, greedy_trust

torch.set_num_threads(2)
OUT = Path("results")
OUT.mkdir(exist_ok=True)
SEEDS = [11, 12, 13]
LAM, ITERS = 1.0, 4000
BASE = dict(nveh=50, rate_=0.7)
ATTACK = dict(black_hole=True, false_reports=True)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def evaluate(policy, seeds=SEEDS, obs_use_dt=True, **env_kw):
    rows = []
    for s in seeds:
        env = VECEnv(seed=s, **env_kw)
        env.obs_use_dt = obs_use_dt
        rows.append(env.run(policy, 60.0))
    keys = rows[0].keys()
    return {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))] for k in keys}


# ---------- data and models ----------
log("collecting expert data")
O, A, MAL = collect(episodes=30, seed=100)
log("samples", len(A), "with a malicious slot:", int(MAL.any(1).sum()))
models = {}
for rho in [0.0, 0.1, 0.2, 0.3]:
    Ap, nhit = poison(A, MAL, rho, seed=7)
    log(f"rho={rho}: {nhit} poisoned labels; training")
    models[rho] = dict(
        tg=DiffusionPolicy(use_trust_input=True, guidance=LAM, steps=10, seed=1).fit(O, Ap, ITERS),
        diff=DiffusionPolicy(use_trust_input=False, guidance=0.0, steps=10, seed=1).fit(O, Ap, ITERS),
        bc=BCMLP(seed=1).fit(O, Ap, ITERS),
        nhit=nhit,
    )
# ablation without digital twin: dataset whose trust features ignore the twin
log("collecting no-DT data for the ablation")


def collect_nodt():
    from tgdiff import data as D
    from tgdiff import env as E
    orig = E.VECEnv.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        self.obs_use_dt = False
    E.VECEnv.__init__ = init
    try:
        return D.collect(episodes=30, seed=100)
    finally:
        E.VECEnv.__init__ = orig


O2, A2, MAL2 = collect_nodt()
A2p, _ = poison(A2, MAL2, 0.2, seed=7)
tg_nodt = DiffusionPolicy(use_trust_input=True, guidance=LAM, use_dt=False, steps=10, seed=1).fit(O2, A2p, ITERS)

R = {"meta": dict(samples=int(len(A)), seeds=SEEDS, lam=LAM, iters=ITERS, base=BASE)}


def policy_set(rho):
    m = models[rho]
    return {"Greedy": greedy, "Greedy+Trust": greedy_trust, "BC-MLP": m["bc"], "Diff": m["diff"], "TG-Diff": m["tg"], "Expert": expert}


# E1: no attack, varying number of vehicles
log("E1")
R["E1"] = {}
for nv in [20, 40, 60, 80]:
    R["E1"][nv] = {k: evaluate(p, nveh=nv, rate_=BASE["rate_"]) for k, p in policy_set(0.0).items()}

# E2: malicious fraction (black-hole RSU + malicious helpers + false reports), clean data
log("E2")
R["E2"] = {}
for mf in [0.0, 0.1, 0.2, 0.3, 0.4]:
    kw = dict(BASE, mal_frac=mf, **(ATTACK if mf > 0 else {}))
    R["E2"][mf] = {k: evaluate(p, **kw) for k, p in policy_set(0.0).items()}

# E3: data poisoning rate, attack at mal_frac = 0.3
log("E3")
R["E3"] = {}
for rho in [0.0, 0.1, 0.2, 0.3]:
    kw = dict(BASE, mal_frac=0.3, **ATTACK)
    ps = {k: v for k, v in policy_set(rho).items() if k in ("BC-MLP", "Diff", "TG-Diff")}
    R["E3"][rho] = {k: evaluate(p, **kw) for k, p in ps.items()}
    R["E3"][rho]["poisoned_labels"] = models[rho]["nhit"]

# E4: number of sampling steps K (attack, rho = 0)
log("E4")
R["E4"] = {}
kw = dict(BASE, mal_frac=0.3, **ATTACK)
for K in [1, 2, 5, 10, 20]:
    R["E4"][K] = {}
    for name in ("diff", "tg"):
        p = models[0.0][name]
        p.steps = K
        R["E4"][K]["TG-Diff" if name == "tg" else "Diff"] = evaluate(p, **kw)
        p.steps = 10

# E5: ablation (attack mal_frac = 0.3, poisoning rho = 0.2)
log("E5")
m = models[0.2]
kw = dict(BASE, mal_frac=0.3, **ATTACK)
noin_guided = DiffusionPolicy(use_trust_input=False, guidance=LAM, steps=10, seed=1)
noin_guided.model = m["diff"].model
in_unguided = DiffusionPolicy(use_trust_input=True, guidance=0.0, steps=10, seed=1)
in_unguided.model = m["tg"].model
R["E5"] = {
    "TG-Diff (full)": evaluate(m["tg"], **kw),
    "w/o trust input": evaluate(noin_guided, **kw),
    "w/o trust guidance": evaluate(in_unguided, **kw),
    "w/o digital twin": evaluate(tg_nodt, obs_use_dt=False, **kw),
    "Diff (none)": evaluate(m["diff"], **kw),
}

json.dump(R, open(OUT / "results.json", "w"), indent=1)
log("done")
