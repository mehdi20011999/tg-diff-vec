# TG-Diff prototype

Prototype code for the paper draft **"Digital Twin-Assisted Trust-Guided Diffusion Policy for Secure Collaborative V2V/V2I Computation Offloading"** (article 2 of the thesis).

> Preliminary prototype. The road model is a synthetic 1.5 km highway (same model as the 3D simulator `genai-vec-offloading-lab`). The full campaign will use SUMO mobility traces.

## Contents

```
tgdiff/env.py        VEC environment: RSUs + MEC, cloud, helper vehicles (V2V), digital twin, beta trust,
                     attacks A1 black-hole RSU, A2 malicious helpers, A3 false state reports
tgdiff/policies.py   Greedy, Greedy+Trust, privileged expert, BC-MLP, diffusion policy (DDPM training,
                     DDIM sampling) with optional trust input and trust guidance (TG-Diff)
tgdiff/data.py       Offline dataset from the privileged expert + attack A4 (label poisoning)
run_experiments.py   Experiments E1-E5 -> results/results.json
make_figures.py      Figures for the paper -> paper/figures/
paper/               LaTeX draft (IEEEtran)
```

## Run

```bash
pip install torch numpy matplotlib
python run_experiments.py      # about 15-20 min on a 2-core CPU
python make_figures.py
```

On Google Colab: upload the folder, run the two commands in a cell with `!`.

## What each experiment shows

| Exp. | Varies | Question |
|---|---|---|
| E1 | number of vehicles, no attack | does protection cost anything without attacks? |
| E2 | share of malicious helpers (+ black-hole RSU, false reports) | robustness to malicious nodes (RQ1, RQ2) |
| E3 | share of poisoned training labels | robustness of the generative policy to data poisoning (RQ1) |
| E4 | sampling steps K | decision quality versus decision time (RQ3) |
| E5 | ablation: trust input, trust guidance, digital twin | which part of TG-Diff matters (RQ2) |

## Project page (GitHub Pages)

The `docs/` folder is a ready-made project page (method, interactive results, roadmap).
On GitHub: **Settings > Pages > Deploy from a branch > main / docs > Save**.
The page loads `docs/data/results.json`; after re-running the experiments, refresh it with:

```bash
python -c "import json; R=json.load(open('results/results.json')); R['E6']=json.load(open('results/results_e6.json')); json.dump(R, open('docs/data/results.json','w'))"
cp paper/figures/*.png docs/img/
```

## Licence

MIT. Copyright (c) 2026 Mehdi Ammou.
