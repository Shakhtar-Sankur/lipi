"""The M1 chart from results/m1/kaggle-full-2026-10-09.txt's numbers (results/m1/m1.png)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "m1", "m1.png")
SURFACE, INK, INK2, GRID, MUTED, ACCENT = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#c9c8c3", "#2a78d6"
plt.rcParams.update({"font.family": "Noto Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})
names = ["Qwen 2.5 0.5B\nas shipped", "A: Qwen tokenizer,\ntrained", "B: extended,\nsame text as A", "C: extended,\nsame GPU time as A"]
odia_web = [1.1295, 0.5749, 0.6078, 0.4609]
minutes = [None, 103.5, 19.3, 108.0]
text_mb = [None, 24.0, 24.0, 135.3]
colors = [MUTED, MUTED, ACCENT, ACCENT]

fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"wspace": 0.45})
for ax in (a, b):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
y = range(len(names))[::-1]
a.barh(list(y), odia_web, color=colors, height=0.62)
for yi, v in zip(y, odia_web):
    a.text(v + 0.015, yi, f"{v:.2f}", va="center", fontsize=10)
a.set_yticks(list(y), names, fontsize=9.5)
a.set_xlim(0, 1.3)
a.set_title("Odia bits per byte on held-out web text\n(lower is better)", loc="left", fontsize=11, fontweight="bold")
yb = [2, 1, 0]
b.barh(yb, minutes[1:], color=colors[1:], height=0.62)
for yi, v, mb in zip(yb, minutes[1:], text_mb[1:]):
    b.text(v + 1.5, yi, f"{v:.0f} min  ({mb:.0f} MB of Odia)", va="center", fontsize=10)
b.set_yticks(yb, names[1:], fontsize=9.5)
b.set_xlim(0, 170)
b.set_title("Training time on Kaggle's 2× T4\n(and how much Odia text it read)", loc="left", fontsize=11, fontweight="bold")
fig.text(0.01, 0.01, "lipi M1, one run per arm. Extended = Qwen's tokenizer + the marks fix + 16K Odia tokens; "
         "8.6x fewer tokens for the same Odia text.", fontsize=8.5, color=INK2)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
