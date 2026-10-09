"""M1, tokenizer half: what each part of the fix does to Qwen 2.5's tokenizer on Odia.

For the base tokenizer, the marks fix alone, new Odia merges alone (n = 1K ... 32K) and both,
it measures the token tax on FLORES-200 devtest (held out: the merges are learned on
FineWeb-2 web text) and checks that every other language's ids are unchanged by the new
merges. Writes results/m1/tokenizer.json and the extended tokenizer M1 trains with
(tokenizers/qwen25-odia-marks-vocab-16k.json).

    python scripts/extend_tokenizer.py [docs]       # docs: training documents used (default 20000)
"""
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tokenizers import Tokenizer  # noqa: E402

from lipi import corpus, extend, flores, languages, tokenizers  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "m1")
BASE = ("Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987")
SIZES = [1000, 4000, 16000, 32000]
SAVE = 16000


def base_cfg():
    path = tokenizers._download(f"https://huggingface.co/{BASE[0]}/resolve/{BASE[1]}/tokenizer.json",
                                os.path.join(tokenizers.CACHE, "tokenizers", "qwen25", "tokenizer.json"))
    return json.load(open(path, encoding="utf-8"))


def count(cfg, texts):
    tok = Tokenizer.from_str(json.dumps(cfg))
    return [e.ids for e in tok.encode_batch(texts, add_special_tokens=False)]


def main():
    n_docs = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    os.makedirs(os.path.join(OUT, "tokenizers"), exist_ok=True)
    base = base_cfg()
    marks = extend.with_marks(base)
    test = {l.code: flores.sentences(l.code) for l in languages.ALL}
    base_ids = {c: count(base, t) for c, t in test.items()}
    eng = sum(map(len, base_ids["eng_Latn"]))
    docs = list(itertools.islice(corpus.documents("ory_Orya"), n_docs))
    rows, t0 = [], time.time()

    def record(name, cfg, added):
        ids = {c: count(cfg, t) for c, t in test.items()}
        same = [c for c in test if c != "ory_Orya" and ids[c] == base_ids[c]]
        r = {"tokenizer": name, "new_tokens": len(added),
             "ory_tax": round(sum(map(len, ids["ory_Orya"])) / eng, 3),
             "eng_tokens_change": sum(map(len, ids["eng_Latn"])) - eng,
             "other_languages_identical": len(same),
             "tax": {c: round(sum(map(len, ids[c])) / eng, 3) for c in test}}
        rows.append(r)
        print(f"{name:24s} new {len(added):6d}  Odia tax {r['ory_tax']:6.2f}  other languages with identical ids "
              f"{len(same)}/{len(test) - 1}  English token change {r['eng_tokens_change']:+d}  ({time.time() - t0:.0f}s)", flush=True)

    record("base", base, [])
    record("marks", marks, [])
    for name, cfg in (("vocab", base), ("marks+vocab", marks)):
        counts = extend.words(Tokenizer.from_str(json.dumps(cfg)), docs, extend.odia)
        learned = extend.learn(cfg, counts, int(max(SIZES) * 1.25))
        merges = extend.confine(learned, cfg["model"]["vocab"], extend.odia_bytes)
        print(f"{name}: {len(learned)} merges learned, {len(merges)} confined to Odia", flush=True)
        for n in SIZES:
            ext, added = extend.apply(cfg, merges[:n])
            record(f"{name} {n // 1000}K", ext, added)
            if n == SAVE and name == "marks+vocab":
                json.dump(ext, open(os.path.join(OUT, "tokenizers", f"qwen25-odia-{name.replace('+', '-')}-{n // 1000}k.json"), "w"),
                          ensure_ascii=False)
    json.dump({"base": BASE, "train_docs": n_docs, "train_bytes": sum(len(d.encode()) for d in docs),
               "english_tokens": eng, "rows": rows}, open(os.path.join(OUT, "tokenizer.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
