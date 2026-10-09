"""Summarises results/m2/speed.json: per language, the time to write the same sentences with
each tokenizer, and how much text fits in a 32K context.

    python scripts/summarize_m2.py [speed.json]
"""
import json
import os
import sys

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "m2", "speed.json")
d = json.load(open(path))
rows = [r for r in d["rows"] if "lang" in r]
toks = list(dict.fromkeys(r["tokenizer"] for r in rows))
langs = list(dict.fromkeys(r["lang"] for r in rows))
by = {(r["tokenizer"], r["lang"]): r for r in rows}
print(f"{d['device']}, {d['sentences']} FLORES devtest sentences per language, written token by token (forced)")
print(f"{'':10s}" + "".join(f"{t:>30s}" for t in toks))
for code in langs:
    cells = []
    for t in toks:
        r = by[(t, code)]
        cells.append(f"{r['seconds_per_sentence']:.3f} s/sent ({r['ms_per_token']:.1f} ms/tok)")
    print(f"{code:10s}" + "".join(f"{c:>30s}" for c in cells))
print("speed-up over Qwen's tokenizer, batch 1 and batch 16:")
for code in langs:
    q = by[("qwen", code)]
    print(f"  {code:10s}" + "".join(f"  {t}: {q['seconds_batch1'] / by[(t, code)]['seconds_batch1']:.2f}x / {q['seconds_batch16'] / by[(t, code)]['seconds_batch16']:.2f}x"
                                      for t in toks if t != "qwen"))
print("FLORES devtest sentences that fit in a 32,768-token context:")
for code in langs:
    print(f"  {code:10s}" + "".join(f"  {t}: {by[(t, code)]['sentences_in_32k_context']}" for t in toks))
for r in d["rows"]:
    if "prefill_odia_3000_chars" in r:
        p = r["prefill_odia_3000_chars"]
        print(f"prefill, 3,000 characters of Odia, {r['tokenizer']}: {p['tokens']} tokens, {1000 * p['seconds']:.1f} ms")
