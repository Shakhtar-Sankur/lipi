"""Quality and speed of a model with a given tokenizer, measured so that tokenizers compare:

- bits per byte on held-out text (FLORES-200 devtest; FineWeb-2 documents never trained on);
- Belebele (Meta, CC BY-SA 4.0): 900 multiple-choice reading-comprehension questions on FLORES
  passages, scored zero-shot by the log-probability of each answer per byte of the answer;
- translation, English -> Odia and Odia -> English, 5-shot from FLORES dev, greedy, on FLORES
  devtest sentences: chrF++ (sacrebleu), the tokens generated, and the seconds it took.
"""
import json
import math
import os
import time
import urllib.request

import torch

CACHE = os.environ.get("LIPI_CACHE", os.path.expanduser("~/.cache/lipi"))
BELEBELE = "https://huggingface.co/datasets/facebook/belebele/resolve/main/data/{code}.jsonl"
NAMES = {"eng_Latn": "English", "ory_Orya": "Odia"}


def belebele(code):
    path = os.path.join(CACHE, "belebele", f"{code}.jsonl")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(BELEBELE.format(code=code), path + ".part")
        os.replace(path + ".part", path)
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def _batches(seqs, batch, pad):
    for k in range(0, len(seqs), batch):
        chunk = seqs[k:k + batch]
        n = max(map(len, chunk))
        ids = torch.full((len(chunk), n), pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), n), dtype=torch.long)
        for j, s in enumerate(chunk):
            ids[j, :len(s)] = torch.tensor(s)
            mask[j, :len(s)] = 1
        yield k, ids, mask


@torch.no_grad()
def continuation_logprobs(model, prefixes, continuations, pad, batch=8, device="cpu"):
    """log p(continuation | prefix) for each pair of token-id lists. The output layer runs
    only on the continuation's positions: full logits for a batch of 2,000-token prompts over
    a 167K vocabulary would not fit in a T4's memory."""
    seqs = [p + c for p, c in zip(prefixes, continuations)]
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = [0.0] * len(seqs)
    decoder, head = model.get_decoder(), model.get_output_embeddings()
    for k, ids, mask in _batches([seqs[i] for i in order], batch, pad):
        ids, mask = ids.to(device), mask.to(device)
        hidden = decoder(input_ids=ids, attention_mask=mask).last_hidden_state
        for j in range(ids.shape[0]):
            i = order[k + j]
            a, b = len(prefixes[i]), len(seqs[i])
            logp = torch.log_softmax(head(hidden[j, a - 1:b - 1]).float(), -1)
            out[i] = logp.gather(-1, ids[j, a:b, None]).sum().item()
    return out


def belebele_accuracy(model, tok, rows, start_id, batch=8, device="cpu"):
    """Zero-shot accuracy: the answer with the highest log-probability per byte."""
    prefixes, conts, nbytes = [], [], []
    for r in rows:
        prompt = f"Passage: {r['flores_passage']}\nQuestion: {r['question']}\nAnswer:"
        p = [start_id] + tok.encode(prompt, add_special_tokens=False).ids
        for k in range(1, 5):
            ans = " " + r[f"mc_answer{k}"]
            prefixes.append(p)
            conts.append(tok.encode(ans, add_special_tokens=False).ids)
            nbytes.append(len(ans.encode()))
    t0 = time.time()
    lp = continuation_logprobs(model, prefixes, conts, start_id, batch, device)
    correct = 0
    for q, r in enumerate(rows):
        scores = [lp[4 * q + k] / nbytes[4 * q + k] for k in range(4)]
        correct += int(1 + max(range(4), key=scores.__getitem__) == int(r["correct_answer_num"]))
    return {"accuracy": correct / len(rows), "questions": len(rows), "seconds": round(time.time() - t0, 1),
            "tokens": sum(len(p) + len(c) for p, c in zip(prefixes, conts))}


def stop_ids(tok):
    """Ids of tokens whose text contains a newline: translation stops at the end of the line."""
    return [i for t, i in tok.get_vocab().items() if "\n" in tok.decode([i])]


@torch.no_grad()
def translate(model, tok, sources, refs, shots, src, tgt, start_id, batch=8, device="cpu"):
    """Greedy few-shot translation; chrF++, tokens generated and seconds."""
    import sacrebleu
    head = "".join(f"{NAMES[src]}: {s}\n{NAMES[tgt]}: {t}\n\n" for s, t in shots)
    prompts = [[start_id] + tok.encode(f"{head}{NAMES[src]}: {s}\n{NAMES[tgt]}:", add_special_tokens=False).ids for s in sources]
    ref_len = [len(tok.encode(" " + r, add_special_tokens=False).ids) for r in refs]
    stops = stop_ids(tok)
    outputs, generated, seconds = [""] * len(sources), 0, 0.0
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for k in range(0, len(order), batch):
        idx = order[k:k + batch]
        n = max(len(prompts[i]) for i in idx)
        ids = torch.full((len(idx), n), start_id, dtype=torch.long)
        mask = torch.zeros((len(idx), n), dtype=torch.long)
        for j, i in enumerate(idx):                                  # left padding
            ids[j, n - len(prompts[i]):] = torch.tensor(prompts[i])
            mask[j, n - len(prompts[i]):] = 1
        limit = int(1.5 * max(ref_len[i] for i in idx)) + 8
        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.time()
        out = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device), max_new_tokens=limit,
                             do_sample=False, eos_token_id=stops, pad_token_id=start_id)
        if device != "cpu":
            torch.cuda.synchronize()
        seconds += time.time() - t0
        for j, i in enumerate(idx):
            new = out[j, n:].tolist()
            cut = next((p for p, t in enumerate(new) if t in stops or t == start_id), len(new))
            generated += cut
            outputs[i] = tok.decode(new[:cut + 1]).split("\n")[0].strip()
    chrf = sacrebleu.corpus_chrf(outputs, [refs], word_order=2).score
    return {"chrf++": round(chrf, 2), "sentences": len(sources), "tokens_generated": generated,
            "seconds": round(seconds, 1), "outputs": outputs}


def bits_per_byte(model, tok, texts, start_id, batch=8, device="cpu"):
    """-log2 p(text) per UTF-8 byte over `texts`, each starting after `start_id`."""
    seqs = [[start_id] + tok.encode(t, add_special_tokens=False).ids for t in texts]
    lp = continuation_logprobs(model, [s[:1] for s in seqs], [s[1:] for s in seqs], start_id, batch, device)
    nbytes = sum(len(t.encode()) for t in texts)
    return {"bits_per_byte": round(-sum(lp) / math.log(2) / nbytes, 4), "tokens": sum(len(s) - 1 for s in seqs),
            "bytes": nbytes}
