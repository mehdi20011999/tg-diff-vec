"""Figures and LaTeX snippets from results/results.json."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("results/results.json"))
FIG = Path("paper/figures")
FIG.mkdir(parents=True, exist_ok=True)
STY = {  # identity: colour + marker + dash, so the figure reads in grey too
    "TG-Diff": dict(color="#2a78d6", marker="o", lw=2.2, ls="-"),
    "Diff": dict(color="#eb6834", marker="s", lw=1.5, ls="-"),
    "BC-MLP": dict(color="#1baf7a", marker="^", lw=1.5, ls="-"),
    "Greedy": dict(color="#eda100", marker="v", lw=1.5, ls="--"),
    "Greedy+Trust": dict(color="#e87ba4", marker="D", lw=1.5, ls="--"),
    "Expert": dict(color="#8a8a85", marker="", lw=1.2, ls=":"),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.labelcolor": INK, "legend.frameon": False})


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def lines(ax, block, xs, names, metric="succ", scale=100):
    for n in names:
        ys = [block[str(x)][n][metric][0] * scale for x in xs]
        es = [block[str(x)][n][metric][1] * scale for x in xs]
        ax.errorbar(xs, ys, yerr=es, capsize=2, ms=4, label=n, **STY[n])


# Fig. A: success vs malicious fraction
fig, ax = plt.subplots(figsize=(3.5, 2.4))
xs = [0.0, 0.1, 0.2, 0.3, 0.4]
lines(ax, R["E2"], xs, ["Expert", "Greedy", "Greedy+Trust", "BC-MLP", "Diff", "TG-Diff"])
ax.set_xticks(xs, [f"{int(x*100)}" for x in xs]); ax.set_xlabel("Malicious helper vehicles (%)")
ax.set_ylabel("Tasks on time (%)"); clean(ax); ax.legend(fontsize=6.5, ncol=2, loc="lower left")
fig.tight_layout(); fig.savefig(FIG / "fig_malicious.pdf"); fig.savefig(FIG / "fig_malicious.png", dpi=200)

# Fig. B: success vs poisoning rate
fig, ax = plt.subplots(figsize=(3.5, 2.3))
xs = [0.0, 0.1, 0.2, 0.3]
lines(ax, R["E3"], xs, ["BC-MLP", "Diff", "TG-Diff"])
ax.set_xticks(xs, [f"{int(x*100)}" for x in xs]); ax.set_xlabel("Poisoned training labels (%)")
ax.set_ylabel("Tasks on time (%)"); clean(ax); ax.legend(fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig(FIG / "fig_poison.pdf"); fig.savefig(FIG / "fig_poison.png", dpi=200)

# Fig. C: steps K, two panels sharing x (no dual axis)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.5, 2.0))
Ks = [1, 2, 5, 10, 20]
for n in ["Diff", "TG-Diff"]:
    a1.plot(Ks, [R["E4"][str(k)][n]["succ"][0] * 100 for k in Ks], ms=4, label=n, **STY[n])
    a2.plot(Ks, [R["E4"][str(k)][n]["over"][0] * 1000 for k in Ks], ms=4, label=n, **STY[n])
for a in (a1, a2):
    a.set_xscale("log"); a.set_xticks(Ks, [str(k) for k in Ks]); a.minorticks_off(); a.set_xlabel("Sampling steps K"); clean(a)
a1.set_ylabel("Tasks on time (%)"); a2.set_ylabel("Decision time (ms)")
a1.legend(fontsize=6.5, loc="best")
fig.tight_layout(); fig.savefig(FIG / "fig_steps.pdf"); fig.savefig(FIG / "fig_steps.png", dpi=200)
print("figures written")
