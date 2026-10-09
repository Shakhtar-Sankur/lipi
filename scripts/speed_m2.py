"""M2: what the token tax costs in time, measured on a GPU.

The model writes each FLORES devtest sentence token by token with its key-value cache, as it
would when answering, but forced to the reference sentence (so every tokenizer writes exactly
the same text, whatever the model's quality). Time per sentence, at batch 1 and batch 16,
for Qwen 2.5 0.5B with Qwen's tokenizer and with lipi's. Speed does not depend on the
weights, so the extended models are the untrained ones (new rows = means of their pieces).

Also: the prefill time of a long document, and how much text fits in Qwen 2.5's 32,768-token
context window.

    python scripts/speed_m2.py [--sentences 40] [--quick]      # writes results/m2/speed.json
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tokenizers import Tokenizer  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from lipi import corpus, flores, model as M  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BASE = os.path.join(corpus.CACHE, "models", "qwen25-0.5b")
TOKENIZERS = {"qwen": None,
              "lipi-odia-16k": os.path.join(ROOT, "results", "m1", "tokenizers", "qwen25-odia-marks-vocab-16k.json"),
              "lipi-indic-96k": os.path.join(ROOT, "results", "m1", "tokenizers", "qwen25-indic-96k.json")}
LANGS = ["eng_Latn", "hin_Deva", "ben_Beng", "tam_Taml", "tel_Telu", "ory_Orya", "urd_Arab", "sat_Olck"]
CONTEXT = 32768


def sync(device):
    if device != "cpu":
        torch.cuda.synchronize()


@torch.inference_mode()
def forced_decode(model, seqs, device):
    """Seconds to run every position of `seqs` (equal-length token lists) one step at a time
    with the key-value cache, as generation does."""
    ids = torch.tensor(seqs, device=device)
    sync(device)
    t0 = time.time()
    out = model(input_ids=ids[:, :1], use_cache=True)
    for k in range(1, ids.shape[1]):
        out = model(input_ids=ids[:, k:k + 1], past_key_values=out.past_key_values, use_cache=True)
    sync(device)
    return time.time() - t0


@torch.inference_mode()
def prefill(model, seq, device, repeats=3):
    ids = torch.tensor([seq], device=device)
    model(input_ids=ids[:, :16])
    best = float("inf")
    for _ in range(repeats):
        sync(device)
        t0 = time.time()
        model(input_ids=ids)
        sync(device)
        best = min(best, time.time() - t0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--quick", action="store_true", help="2 sentences of English and Odia: checks the script, not speed")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "m2", "speed.json"))
    a = ap.parse_args()
    n = 2 if a.quick else a.sentences
    langs = ["eng_Latn", "ory_Orya"] if a.quick else LANGS
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32
    base_cfg = json.load(open(os.path.join(BASE, "tokenizer.json"), encoding="utf-8"))
    doc = next(d for d in corpus.documents("ory_Orya", "heldout") if len(d) > 3000)[:3000]
    results = {"device": torch.cuda.get_device_name(0) if device != "cpu" else "cpu", "sentences": n, "rows": []}
    for name, path in TOKENIZERS.items():
        cfg = base_cfg if path is None else json.load(open(path, encoding="utf-8"))
        tok = Tokenizer.from_str(json.dumps(cfg))
        model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
        if path is not None:
            M.extend_embeddings(model, base_cfg, cfg)
        model.to(dtype).to(device).eval()
        start = tok.token_to_id("<|endoftext|>")
        forced_decode(model, [[start] * 8], device)                                    # warm up
        for code in langs:
            texts = flores.sentences(code)[:n]
            seqs = [[start] + tok.encode(t, add_special_tokens=False).ids for t in texts]
            one = sum(forced_decode(model, [s], device) for s in seqs)
            tokens = sum(len(s) - 1 for s in seqs)
            # batch: sentences cut to the shortest in each batch would change the text, so pad to the
            # longest and count only real tokens (padding costs time, as it would in a server)
            batched = 0.0
            for k in range(0, len(seqs), a.batch):
                chunk = seqs[k:k + a.batch]
                m = max(map(len, chunk))
                batched += forced_decode(model, [s + [start] * (m - len(s)) for s in chunk], device)
            all_tokens = [len(tok.encode(t, add_special_tokens=False).ids) for t in flores.sentences(code)]
            fit = 0
            used = 0
            for t in all_tokens:
                if used + t > CONTEXT:
                    break
                used += t
                fit += 1
            row = {"tokenizer": name, "lang": code, "sentences": n, "tokens": tokens,
                   "seconds_batch1": round(one, 3), "ms_per_token": round(1000 * one / tokens, 2),
                   "seconds_per_sentence": round(one / n, 4), "seconds_batch16": round(batched, 3),
                   "flores_tokens_all": sum(all_tokens),
                   "sentences_in_32k_context": fit if fit < len(all_tokens) else f">{len(all_tokens)}"}
            results["rows"].append(row)
            print(json.dumps(row), flush=True)
        dseq = [start] + tok.encode(doc, add_special_tokens=False).ids
        if len(dseq) <= CONTEXT:
            p = prefill(model, dseq, device)
            row = {"tokenizer": name, "prefill_odia_3000_chars": {"tokens": len(dseq) - 1, "seconds": round(p, 4)}}
            results["rows"].append(row)
            print(json.dumps(row), flush=True)
        del model
        if device != "cpu":
            torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
