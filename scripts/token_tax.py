"""M0: the token tax. Every tokenizer in lipi.tokenizers on the same 1,012 FLORES-200 devtest
sentences in English and every Indian language in FLORES-200.

    python scripts/token_tax.py            # writes results/m0/token_tax.json and .csv
"""
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lipi import flores, languages, measure, tokenizers  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "m0")


def run_one(key):
    t0 = time.time()
    tok = tokenizers.load(key)
    rows = {}
    for lang in languages.ALL:
        rows[lang.code] = measure.corpus(tok, flores.sentences(lang.code))
    return key, {"style": tok.style, "vocab": tok.vocab_size, "seconds": round(time.time() - t0, 1), "langs": rows}


def derive(raw):
    out = []
    for key, r in raw.items():
        spec = tokenizers.BY_KEY[key]
        eng = r["langs"]["eng_Latn"]["tokens"]
        for code, c in r["langs"].items():
            lang = languages.BY_CODE[code]
            row = {"tokenizer": key, "label": spec.label, "used_by": spec.used_by, "vocab": r["vocab"],
                   "lang": code, "language": lang.name, "script": lang.script, "scheduled": lang.scheduled,
                   "speakers_m": lang.speakers, "sentences": c["sentences"], "tokens": c["tokens"],
                   "tax": round(c["tokens"] / eng, 3),
                   "tokens_per_letter": round(c["tokens"] / c["letters"], 3),
                   "chars_per_token": round(c["chars"] / c["tokens"], 3),
                   "unk_tokens": c["unk"], "exact_sentences": c["exact"]}
            if "floor" in c:
                row["floor_tax"] = round(c["floor"] / eng, 3)
            if c["exact"]:
                row["broken_letters"] = round(c["broken"] / c["exact_letters"], 4)
                row["byte_fragments"] = round(c["fragments"] / c["exact_tokens"], 4)
            out.append(row)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    keys = [s.key for s in tokenizers.SPECS]
    for k in keys:                       # download once, before the workers start
        tokenizers.load(k)
    flores.root()
    with ProcessPoolExecutor(4) as ex:
        raw = dict(ex.map(run_one, keys))
    rows = derive(raw)
    json.dump({"source": "FLORES-200 devtest, 1,012 sentences per language", "raw": raw, "rows": rows},
              open(os.path.join(OUT, "token_tax.json"), "w"), ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "token_tax.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    by = {(r["tokenizer"], r["lang"]): r for r in rows}
    langs = [l.code for l in languages.INDIC]
    print(f"{'language':22s}" + "".join(f"{tokenizers.BY_KEY[k].label[:9]:>10s}" for k in keys))
    for code in langs:
        print(f"{languages.BY_CODE[code].name:22s}" + "".join(f"{by[(k, code)]['tax']:10.2f}" for k in keys))


if __name__ == "__main__":
    main()
