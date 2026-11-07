import numpy as np
from tgdiff.env import VECEnv
SEEDS=[11,12,13]
def evaluate_fn(policy, obs_use_dt=True, **kw):
    rows=[]
    for s in SEEDS:
        env=VECEnv(seed=s, **kw); env.obs_use_dt=obs_use_dt; rows.append(env.run(policy,60.0))
    return {k:[float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))] for k in rows[0]}
