"""lipi-Indic against GPT-4o / GPT-5 and Gemini, per language (results/m1/indic.png), from
results/m0/token_tax.json and results/m1/indic.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SURFACE, INK, INK2, GRID, MUTED, MID, ACCENT = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#c9c8c3", "#8e8d88", "#2a78d6"
plt.rcParams.update({"font.family": "Noto Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})
CAP = 6.0


def main(size="96K"):
    m0 = json.load(open(os.path.join(ROOT, "results", "m0", "token_tax.json")))["rows"]
    m1 = json.load(open(os.path.join(ROOT, "results", "m1", "indic.json")))
    lipi = next(r for r in m1["rows"] if r["tokenizer"] == f"lipi-indic {size}")
    tax = {(r["tokenizer"], r["lang"]): r["tax"] for r in m0}
    langs = sorted({(r["lang"], r["language"], r["speakers_m"]) for r in m0 if r["lang"] != "eng_Latn"}, key=lambda x: -x[2])
    series = [("GPT-4o / GPT-5 (o200k)", MUTED, lambda c: tax[("o200k", c)]),
              ("Gemini (Gemma 3's tokenizer)", MID, lambda c: tax[("gemma3", c)]),
              (f"lipi-Indic {size} (Qwen 2.5 + {size} tokens)", ACCENT, lambda c: lipi["tax"][c])]
    fig, ax = plt.subplots(figsize=(10, 11.5))
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    h = 0.27
    for k, (label, color, f) in enumerate(series):
        ys = [len(langs) - 1 - i + (1 - k) * h for i in range(len(langs))]
        vs = [f(c) for c, _, _ in langs]
        ax.barh(ys, [min(v, CAP) for v in vs], height=h, color=color, label=label)
        for y, v in zip(ys, vs):
            if v > CAP:
                ax.text(CAP + 0.05, y, f"{v:.1f}×", va="center", fontsize=8, color=INK2)
            elif k == 2:
                ax.text(v + 0.05, y, f"{v:.2f}×", va="center", fontsize=8, color=ACCENT, fontweight="bold")
    ax.set_yticks(range(len(langs))[::-1], [f"{n}  ({s:.0f}M)" if s >= 1 else f"{n}  ({s:.2f}M)" if s else n for _, n, s in langs],
                  fontsize=9.5)
    ax.axvline(1, color=INK2, linewidth=0.8, linestyle=":")
    ax.set_xlim(0, CAP + 0.6)
    ax.set_xlabel("tokens for the same 1,012 FLORES-200 sentences, as a multiple of English (lower is cheaper)")
    ax.set_ylim(-0.6, len(langs) - 0.4)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    ax.set_title(f"lipi-Indic against ChatGPT's and Gemini's tokenizers\nspeaker-weighted: GPT-4o "
                 f"{weighted(m0, 'o200k'):.2f}×, Gemini {weighted(m0, 'gemma3'):.2f}×, lipi-Indic {lipi['speaker_weighted']:.2f}×"
                 f" (Census 2011 speakers in brackets)", loc="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "m1", "indic.png"), dpi=150)


def weighted(rows, key):
    rs = [r for r in rows if r["tokenizer"] == key and r["lang"] != "eng_Latn"]
    return sum(r["tax"] * r["speakers_m"] for r in rs) / sum(r["speakers_m"] for r in rs)


if __name__ == "__main__":
    main()
