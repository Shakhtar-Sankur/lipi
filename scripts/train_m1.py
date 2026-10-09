"""M1: continued pre-training of Qwen 2.5 0.5B on Odia (plus some English, so English is not
forgotten), with Qwen's own tokenizer or with the extended one. Same documents, same order,
same hyper-parameters; what differs is how many tokens the text becomes.

    torchrun --nproc_per_node 2 scripts/train_m1.py --tokenizer extended --odia-mb 24 --english-mb 6 --out runs/B
    # --tokens N: instead of a fixed amount of text, read text until N tokens (equal compute)

Writes the model (fp16 safetensors), its tokenizer.json and train.json (tokens, steps, time,
loss curve) into --out.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tokenizers import Tokenizer  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from lipi import corpus, model as M  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EXTENDED = os.path.join(ROOT, "results", "m1", "tokenizers", "qwen25-odia-marks-vocab-16k.json")


def documents(tok, odia_bytes, english_bytes, token_budget, seed):
    """Odia and English documents (4:1 by bytes unless given), tokenized, each ending with
    <|endoftext|>, shuffled; stops at the byte amounts, or at `token_budget` tokens."""
    eos = tok.token_to_id("<|endoftext|>")
    ratio = odia_bytes / max(odia_bytes + english_bytes, 1)
    streams = {"ory_Orya": corpus.documents("ory_Orya"), "eng_Latn": corpus.documents("eng_Latn")}
    used = {"ory_Orya": 0, "eng_Latn": 0}
    toks = {"ory_Orya": 0, "eng_Latn": 0}
    docs = []
    while True:
        if token_budget:
            if sum(toks.values()) >= token_budget:
                break
            code = "ory_Orya" if used["ory_Orya"] <= ratio * max(sum(used.values()), 1) else "eng_Latn"
        else:
            left = [c for c, cap in (("ory_Orya", odia_bytes), ("eng_Latn", english_bytes)) if used[c] < cap]
            if not left:
                break
            code = left[0] if len(left) == 1 else ("ory_Orya" if used["ory_Orya"] / odia_bytes <= used["eng_Latn"] / english_bytes else "eng_Latn")
        text = next(streams[code])
        ids = tok.encode(text, add_special_tokens=False).ids + [eos]
        docs.append(ids)
        used[code] += len(text.encode())
        toks[code] += len(ids)
    random.Random(seed).shuffle(docs)
    return docs, {"bytes": used, "tokens": toks}


def pack(docs, seq):
    flat = [t for d in docs for t in d]
    n = len(flat) // (seq + 1)
    return torch.tensor(flat[:n * (seq + 1)], dtype=torch.long).view(n, seq + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", choices=["qwen", "extended"], required=True)
    ap.add_argument("--model", default=os.path.join(corpus.CACHE, "models", "qwen25-0.5b"))
    ap.add_argument("--odia-mb", type=float, default=24)
    ap.add_argument("--english-mb", type=float, default=6)
    ap.add_argument("--tokens", type=int, default=0)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--micro", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--embedding-lr", type=float, default=5e-4)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    rank, world = 0, 1
    if distributed:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu"
    if device != "cpu":
        torch.cuda.set_device(device)
    torch.manual_seed(a.seed)

    base_cfg = json.load(open(os.path.join(a.model, "tokenizer.json"), encoding="utf-8"))
    cfg = base_cfg if a.tokenizer == "qwen" else json.load(open(EXTENDED, encoding="utf-8"))
    tok = Tokenizer.from_str(json.dumps(cfg))
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    new_rows = 0
    if a.tokenizer == "extended":
        new_rows = len(M.extend_embeddings(model, base_cfg, cfg))
    model.config.use_cache = False
    if device != "cpu":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)

    docs, data_info = documents(tok, int(a.odia_mb * 2**20), int(a.english_mb * 2**20), a.tokens, a.seed)
    blocks = pack(docs, a.seq)
    blocks = blocks[:len(blocks) // world * world][rank::world]   # every rank gets as many blocks
    per_step = a.micro * a.accum
    steps = len(blocks) // per_step
    if a.max_steps:
        steps = min(steps, a.max_steps)
    emb = model.get_input_embeddings().weight
    others = [p for p in model.parameters() if p is not emb]
    opt = torch.optim.AdamW([{"params": [emb], "lr": a.embedding_lr}, {"params": others, "lr": a.lr}],
                            betas=(0.9, 0.95), weight_decay=0.0, foreach=False)   # no parameter-sized temporaries
    peaks = [a.embedding_lr, a.lr]
    warm = max(1, steps // 20)
    ddp = LossModel(model)
    if distributed:   # gradient_as_bucket_view: the gradients live in DDP's buckets, not in a second copy
        ddp = torch.nn.parallel.DistributedDataParallel(ddp, device_ids=[torch.device(device)] if device != "cpu" else None,
                                                        gradient_as_bucket_view=True)
    scaler = torch.amp.GradScaler("cuda", enabled=device != "cpu")
    log, t0, seen = [], time.time(), 0
    if rank == 0:
        print(json.dumps({"phase": "data", "tokenizer": a.tokenizer, "new_rows": new_rows, **data_info,
                          "blocks_per_rank": len(blocks), "steps": steps, "tokens_per_step": per_step * a.seq * world}), flush=True)
    for step in range(steps):
        frac = step / max(steps, 1)
        scale = (step + 1) / warm if step < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(steps - warm, 1)))
        for g, peak in zip(opt.param_groups, peaks):
            g["lr"] = peak * scale
        total = 0.0
        for k in range(a.accum):
            batch = blocks[(step * per_step + k * a.micro):(step * per_step + (k + 1) * a.micro)].to(device)
            sync = k == a.accum - 1 or not distributed
            ctx = ddp.no_sync() if (distributed and not sync) else torch.enable_grad()
            with ctx, torch.autocast("cuda", dtype=torch.float16, enabled=device != "cpu"):
                loss = ddp(batch)
                scaler.scale(loss / a.accum).backward()
            total += loss.item() / a.accum
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        seen += per_step * a.seq * world
        if rank == 0 and (step % 20 == 0 or step == steps - 1):
            el = time.time() - t0
            rec = {"step": step, "loss": round(total, 4), "tokens": seen, "elapsed": round(el, 1),
                   "tokens_per_s": round(seen / max(el, 1e-9)), "progress": round(frac, 3)}
            if device != "cpu":
                rec["max_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            log.append(rec)
            print(json.dumps(rec), flush=True)
    if rank == 0:
        os.makedirs(a.out, exist_ok=True)
        model.to(torch.float16).save_pretrained(a.out, safe_serialization=True)
        json.dump(cfg, open(os.path.join(a.out, "tokenizer.json"), "w"), ensure_ascii=False)
        json.dump({"args": vars(a), **data_info, "steps": steps, "tokens_trained": seen, "seconds": round(time.time() - t0, 1),
                   "world": world, "log": log}, open(os.path.join(a.out, "train.json"), "w"), indent=1)
    if distributed:
        dist.destroy_process_group()


class LossModel(torch.nn.Module):
    """Mean next-token cross-entropy over a packed block, without ever holding the whole
    (positions x vocabulary) logit matrix: the output layer and the loss run on chunks of
    positions, each recomputed in the backward pass. On a T4 the full fp32 logits of one
    1,024-token block over a 152K-168K vocabulary (0.6 GB, and as much again for their
    gradient) are what overflows memory next to the weights and AdamW state."""

    def __init__(self, model, chunk=256):
        super().__init__()
        self.model, self.chunk = model, chunk

    def forward(self, batch):
        hidden = self.model.get_decoder()(input_ids=batch[:, :-1]).last_hidden_state
        weight = self.model.get_output_embeddings().weight
        h, y = hidden.reshape(-1, hidden.shape[-1]), batch[:, 1:].reshape(-1)
        total = hidden.new_zeros((), dtype=torch.float32)
        for k in range(0, h.shape[0], self.chunk):
            total = total + torch.utils.checkpoint.checkpoint(_chunk_loss, h[k:k + self.chunk], weight, y[k:k + self.chunk],
                                                              use_reentrant=False)
        return total / h.shape[0]


def _chunk_loss(h, weight, y):
    return torch.nn.functional.cross_entropy((h @ weight.t()).float(), y, reduction="sum")


if __name__ == "__main__":
    main()
