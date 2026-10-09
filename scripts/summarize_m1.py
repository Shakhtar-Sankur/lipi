"""Summarises an M1 run: training cost per arm and every evaluation, one table.

    python scripts/summarize_m1.py runs
"""
import json
import os
import sys

runs = sys.argv[1] if len(sys.argv) > 1 else "runs"
evals = {}
for line in open(os.path.join(runs, "evals.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    evals[r["name"]] = r
order = [n for n in ("base", "init", "A", "B", "C") if n in evals]
for n in order:
    t = os.path.join(runs, n, "train.json")
    if os.path.exists(t):
        j = json.load(open(t))
        evals[n]["train"] = {"tokens": j["tokens_trained"], "seconds": j["seconds"], "steps": j["steps"],
                             "bytes": j["bytes"], "final_loss": j["log"][-1]["loss"] if j["log"] else None}
        print(f"train {n}: {j['tokens_trained']:,} tokens, {j['steps']} steps, {j['seconds'] / 60:.1f} min, "
              f"text {j['bytes']['ory_Orya'] / 2**20:.1f} MB Odia + {j['bytes']['eng_Latn'] / 2**20:.1f} MB English, "
              f"final loss {evals[n]['train']['final_loss']}")
rows = [("Odia bits/byte, FLORES", lambda r: r["bpb_flores_ory_Orya"]["bits_per_byte"]),
        ("Odia bits/byte, web (held out)", lambda r: r["bpb_web_ory_Orya"]["bits_per_byte"]),
        ("English bits/byte, FLORES", lambda r: r["bpb_flores_eng_Latn"]["bits_per_byte"]),
        ("Odia tokens, FLORES", lambda r: r["bpb_flores_ory_Orya"]["tokens"]),
        ("Belebele Odia", lambda r: round(100 * r["belebele_ory_Orya"]["accuracy"], 1)),
        ("Belebele English", lambda r: round(100 * r["belebele_eng_Latn"]["accuracy"], 1)),
        ("Belebele Odia seconds", lambda r: r["belebele_ory_Orya"]["seconds"]),
        ("en->or chrF++", lambda r: r["translate_eng_ory"]["chrf++"]),
        ("en->or tokens generated", lambda r: r["translate_eng_ory"]["tokens_generated"]),
        ("en->or seconds", lambda r: r["translate_eng_ory"]["seconds"]),
        ("or->en chrF++", lambda r: r["translate_ory_eng"]["chrf++"]),
        ("or->en seconds", lambda r: r["translate_ory_eng"]["seconds"])]
print(f"{'':32s}" + "".join(f"{n:>10s}" for n in order))
for label, get in rows:
    print(f"{label:32s}" + "".join(f"{get(evals[n]):>10}" for n in order))
for n in order:
    print(f"{n} en->or sample: {evals[n]['translate_eng_ory']['outputs'][0][:160]}")
for r in evals.values():
    for k in ("translate_eng_ory", "translate_ory_eng"):
        r[k].pop("outputs", None)
    r.pop("model", None)
print("summary-json " + json.dumps(evals, ensure_ascii=False))
