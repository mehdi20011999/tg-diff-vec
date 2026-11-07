"""E6: false state reports only (liars do not drop tasks), heavy load: isolates the value of the digital twin."""
import json, time, numpy as np, torch
from functools import partial
from tgdiff.data import collect
from tgdiff.env import VECEnv
from tgdiff.policies import DiffusionPolicy, greedy, greedy_trust
from run_experiments_common import evaluate_fn
torch.set_num_threads(2)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
O, A, _ = collect(episodes=30, seed=100)
tg = DiffusionPolicy(use_trust_input=True, guidance=1.0, steps=10, seed=1).fit(O, A, 4000)
from tgdiff import env as E
orig = E.VECEnv.__init__
def init(self, *a, **k):
    orig(self, *a, **k); self.obs_use_dt = False
E.VECEnv.__init__ = init
O2, A2, _ = collect(episodes=30, seed=100)
E.VECEnv.__init__ = orig
tg_nodt = DiffusionPolicy(use_trust_input=True, guidance=1.0, use_dt=False, steps=10, seed=1).fit(O2, A2, 4000)
log("trained")
R = {}
for nv in [60, 80]:
    kw = dict(nveh=nv, rate_=1.0, mal_frac=0.3, black_hole=True, false_reports=True, p_rsu=0.0, p_veh=0.0)
    R[nv] = {
        "Greedy": evaluate_fn(greedy, **kw),
        "Greedy+Trust (no DT)": evaluate_fn(partial(greedy_trust, use_dt=False), **kw),
        "Greedy+Trust (DT)": evaluate_fn(greedy_trust, **kw),
        "TG-Diff (no DT)": evaluate_fn(tg_nodt, obs_use_dt=False, **kw),
        "TG-Diff (DT)": evaluate_fn(tg, **kw),
    }
json.dump(R, open("results/results_e6.json", "w"), indent=1)
for nv, b in R.items():
    for n, d in b.items():
        print(nv, n.ljust(22), f"{d['succ'][0]*100:5.1f}±{d['succ'][1]*100:3.1f} lat {d['lat'][0]*1000:4.0f}±{d['lat'][1]*1000:3.0f}")
