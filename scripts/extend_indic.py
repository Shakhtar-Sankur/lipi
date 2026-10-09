"""lipi-Indic: one extension of Qwen 2.5's tokenizer for every Indian language in FLORES-200,
against the tokenizers of GPT-4o / GPT-5 (o200k_base) and Gemini (Gemma 3's, reported by
third-party ports of Gemini's tokenizer to be the same 262,144-entry SentencePiece model).

The marks fix (for the Indian scripts' marks only), then new merges learned on at least 12 MB
of FineWeb-2 web text per language (all of it where a language has less), confined to tokens
that contain an Indian script's characters. Measured on FLORES-200 devtest, which no training
text contains; ten other languages check that the rest of the world's text is untouched.

    python scripts/extend_indic.py      # writes results/m1/indic.json and tokenizers/qwen25-indic-*.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tokenizers import Tokenizer  # noqa: E402

from lipi import corpus, extend, flores, languages  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extend_tokenizer import base_cfg, count  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "m1")
SIZES = [16000, 32000, 64000, 96000, 128000]
SAVE = [64000, 96000]
OTHERS = ["fra_Latn", "spa_Latn", "rus_Cyrl", "zho_Hans", "jpn_Jpan", "kor_Hang", "tha_Thai", "arb_Arab", "pes_Arab", "heb_Hebr"]


def weighted(tax):
    total = sum(l.speakers for l in languages.INDIC)
    return sum(tax[l.code] * l.speakers for l in languages.INDIC) / total


def main():
    t0 = time.time()
    base = base_cfg()
    marks = extend.with_marks(base, extend.INDIC_MARKS)
    codes = [l.code for l in languages.ALL] + OTHERS
    test = {c: flores.sentences(c) for c in codes}
    base_ids = {c: count(base, t) for c, t in test.items()}
    eng = sum(map(len, base_ids["eng_Latn"]))
    tok = Tokenizer.from_str(json.dumps(marks))
    counts, per_lang = None, {}
    for l in languages.INDIC:
        if not os.path.exists(corpus.path(l.code, "tok.jsonl")):
            print(l.code, corpus.fetch_bytes(l.code, 12), flush=True)       # whole file if smaller
        c = extend.words(tok, corpus.documents(l.code, "tok"), extend.indic)
        per_lang[l.code] = sum(c.values())
        counts = c if counts is None else counts + c
    print(f"{len(counts):,} distinct words, {sum(counts.values()):,} in all ({time.time() - t0:.0f}s)", flush=True)
    learned = extend.learn(marks, counts, int(max(SIZES) * 1.1))
    merges = extend.confine(learned, marks["model"]["vocab"], extend.indic_bytes)
    print(f"{len(learned):,} merges learned, {len(merges):,} confined to Indian scripts ({time.time() - t0:.0f}s)", flush=True)
    rows = []

    def record(name, cfg, n_new, ids_ref):
        ids = {c: count(cfg, t) for c, t in test.items()}
        tax = {c: round(sum(map(len, ids[c])) / eng, 3) for c in codes}
        same = [c for c in OTHERS + ["eng_Latn"] if ids[c] == ids_ref[c]]
        r = {"tokenizer": name, "new_tokens": n_new, "vocab": len(cfg["model"]["vocab"]) + len(cfg.get("added_tokens", [])),
             "tax": tax, "speaker_weighted": round(weighted(tax), 3), "others_identical": same}
        rows.append(r)
        print(f"{name:18s} vocab {r['vocab']:7,d}  weighted {r['speaker_weighted']:.3f}  Odia {tax['ory_Orya']:.2f}  Hindi {tax['hin_Deva']:.2f}  "
              f"Tamil {tax['tam_Taml']:.2f}  Santali {tax['sat_Olck']:.2f}  Urdu {tax['urd_Arab']:.2f}  unchanged: {len(same)}/{len(OTHERS) + 1} "
              f"({time.time() - t0:.0f}s)", flush=True)
        return ids

    record("qwen25", base, 0, base_ids)
    record("qwen25+marks", marks, 0, base_ids)
    for n in SIZES:
        ext, added = extend.apply(marks, merges[:n])
        record(f"lipi-indic {n // 1000}K", ext, len(added), base_ids)
        if n in SAVE:
            json.dump(ext, open(os.path.join(OUT, "tokenizers", f"qwen25-indic-{n // 1000}k.json"), "w"), ensure_ascii=False)
    json.dump({"base": "Qwen/Qwen2.5-0.5B", "train_words_per_language": per_lang, "english_tokens": eng,
               "others": OTHERS, "rows": rows}, open(os.path.join(OUT, "indic.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
