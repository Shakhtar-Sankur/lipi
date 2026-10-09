"""M1: evaluate one model with one tokenizer (one JSON line on stdout and in --out).

    python scripts/eval_m1.py --name base --tokenizer qwen                      # Qwen 2.5 0.5B as shipped
    python scripts/eval_m1.py --name init --tokenizer extended                  # extended, new rows = means, no training
    python scripts/eval_m1.py --name B --model runs/B                           # a trained run (its own tokenizer)
    --quick: small subsets, for a smoke test
"""
import argparse
import itertools
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tokenizers import Tokenizer  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from lipi import corpus, evaluate as E, flores, model as M  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BASE = os.path.join(corpus.CACHE, "models", "qwen25-0.5b")
EXTENDED = os.path.join(ROOT, "results", "m1", "tokenizers", "qwen25-odia-marks-vocab-16k.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--tokenizer", choices=["qwen", "extended", "own"], default="own")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype)
    base_cfg = json.load(open(os.path.join(BASE, "tokenizer.json"), encoding="utf-8"))
    if a.tokenizer == "extended":
        cfg = json.load(open(EXTENDED, encoding="utf-8"))
        model.float()
        M.extend_embeddings(model, base_cfg, cfg)
        model.to(dtype)
    elif a.tokenizer == "qwen":
        cfg = base_cfg
    else:
        cfg = json.load(open(os.path.join(a.model, "tokenizer.json"), encoding="utf-8"))
    model.to(device).eval()
    tok = Tokenizer.from_str(json.dumps(cfg))
    start = tok.token_to_id("<|endoftext|>")
    n_sent, n_q, n_tr, n_web = (24, 12, 6, 12) if a.quick else (1012, 900, 200, 500)
    r = {"name": a.name, "model": a.model, "tokenizer": a.tokenizer, "vocab": tok.get_vocab_size()}
    t0 = time.time()
    for code in ("ory_Orya", "eng_Latn"):
        r[f"bpb_flores_{code}"] = E.bits_per_byte(model, tok, flores.sentences(code)[:n_sent], start, a.batch, device)
    web = [d[:2000] for d in itertools.islice(corpus.documents("ory_Orya", "heldout"), n_web)]
    r["bpb_web_ory_Orya"] = E.bits_per_byte(model, tok, web, start, a.batch, device)
    print(json.dumps({"phase": "bpb", "name": a.name, "seconds": round(time.time() - t0)}), flush=True)
    for code in ("ory_Orya", "eng_Latn"):
        r[f"belebele_{code}"] = E.belebele_accuracy(model, tok, E.belebele(code)[:n_q], start, max(1, a.batch // 2), device)
    print(json.dumps({"phase": "belebele", "name": a.name, "seconds": round(time.time() - t0)}), flush=True)
    dev_en, dev_or = flores.sentences("eng_Latn", "dev"), flores.sentences("ory_Orya", "dev")
    test_en, test_or = flores.sentences("eng_Latn")[:n_tr], flores.sentences("ory_Orya")[:n_tr]
    shots = list(zip(dev_en[:5], dev_or[:5]))
    r["translate_eng_ory"] = E.translate(model, tok, test_en, test_or, shots, "eng_Latn", "ory_Orya", start, a.batch, device)
    r["translate_ory_eng"] = E.translate(model, tok, test_or, test_en, [(o, e) for e, o in shots], "ory_Orya", "eng_Latn", start, a.batch, device)
    r["seconds"] = round(time.time() - t0, 1)
    line = json.dumps(r, ensure_ascii=False)
    print(line, flush=True)
    if a.out:
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
