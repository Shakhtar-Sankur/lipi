"""Charts for the README from results/m0/token_tax.json.

    python scripts/charts.py      # writes results/m0/*.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lipi import languages  # noqa: E402

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "m0")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ACCENT, MUTED = "#2a78d6", "#c9c8c3"
# one column per distinct tokenizer (Krutrim-2 and Sarvam-M give Tekken's counts exactly)
COLS = ["o200k", "cl100k", "llama4", "llama3", "gemma3", "qwen3", "deepseek3", "tekken", "mistral3",
        "sarvam1", "bloom", "nllb", "indicbert2"]
COL_LABEL = {"o200k": "GPT-4o", "cl100k": "GPT-4", "llama4": "Llama 4", "llama3": "Llama 3",
             "gemma3": "Gemma 3", "qwen3": "Qwen 3", "deepseek3": "DeepSeek-V3",
             "tekken": "Mistral NeMo\nKrutrim-2\nSarvam-M", "mistral3": "Mistral 7B",
             "sarvam1": "Sarvam-1", "bloom": "BLOOM", "nllb": "NLLB-200", "indicbert2": "IndicBERT v2"}

plt.rcParams.update({"font.family": "Noto Sans", "font.size": 10, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def load():
    rows = json.load(open(os.path.join(HERE, "token_tax.json")))["rows"]
    return {(r["tokenizer"], r["lang"]): r for r in rows}


def heatmap(by):
    langs = languages.INDIC
    data = [[by[(k, l.code)]["tax"] for k in COLS] for l in langs]
    fig, ax = plt.subplots(figsize=(13, 10.5))
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    norm = LogNorm(vmin=1, vmax=14)
    ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    for i, row in enumerate(data):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.1f}" if v < 9.95 else f"{v:.0f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if norm(v) > 0.45 else INK, fontweight="bold" if v >= 5 else "normal")
    ax.set_xticks(range(len(COLS)), [COL_LABEL[k] for k in COLS], fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(langs)), [f"{l.name}" + ("" if l.scheduled else " *") for l in langs], fontsize=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([x - .5 for x in range(1, len(COLS))], minor=True)
    ax.set_yticks([y - .5 for y in range(1, len(langs))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    fig.suptitle("The token tax: tokens for the same 1,012 sentences, as a multiple of English", x=0.02, ha="left",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.text(0.02, 0.012, "FLORES-200 devtest. Languages by number of speakers (Census 2011); * not a scheduled "
             "language. Krutrim-2 and Sarvam-M ship Mistral NeMo's tokenizer (identical counts).",
             fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(os.path.join(HERE, "token_tax_heatmap.png"), dpi=150)
    plt.close(fig)


def bars(by, key, name):
    langs = sorted(languages.INDIC, key=lambda l: by[(key, l.code)]["tax"])
    vals = [by[(key, l.code)]["tax"] for l in langs]
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = [ACCENT if l.code == "ory_Orya" else MUTED for l in langs]
    ax.barh(range(len(langs)), vals, color=colors, height=0.72)
    for i, v in enumerate(vals):
        ax.text(v + 0.12, i, f"{v:.2f}×", va="center", fontsize=9, color=INK)
    ax.axvline(1, color=INK2, linewidth=1)
    ax.text(1.05, len(langs) - 0.3, "English = 1×", fontsize=9, color=INK2)
    ax.set_yticks(range(len(langs)), [l.name for l in langs])
    ax.set_xlim(0, max(vals) * 1.12)
    ax.tick_params(length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlabel("tokens for the same text, as a multiple of English")
    fig.suptitle(f"{name}: what the same text costs in each Indian language", x=0.02, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f"tax_{key}.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    by = load()
    heatmap(by)
    bars(by, "o200k", "GPT-4o")
    bars(by, "llama4", "Llama 4")
