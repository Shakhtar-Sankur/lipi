"""Web text for training, from FineWeb-2 (Hugging Face, ODC-By), which names its subsets with the
same codes as FLORES-200 (ory_Orya, hin_Deva, ...).

Only a sample is read: row groups spread evenly through the language's first file, fetched
over HTTP range requests, so a few hundred MB of text needs no 1.4 GB download. Documents
containing any FLORES-200 sentence (dev or devtest) are dropped, so the evaluation text
never appears in training.
"""
import json
import os

from . import flores

CACHE = os.environ.get("LIPI_CACHE", os.path.expanduser("~/.cache/lipi"))
REPO = "datasets/HuggingFaceFW/fineweb-2/data/{code}/train/000_00000.parquet"


def sample(code, groups, every, min_score=0.9, holdout=False):
    """Documents from `groups` row groups taken every `every` groups (offset by half a stride
    for the held-out sample, so the two never share a row group)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    f = HfFileSystem().open(REPO.format(code=code), "rb", block_size=8 * 2**20)
    pf = pq.ParquetFile(f)
    start = every // 2 if holdout else 0
    picks = list(range(start, pf.metadata.num_row_groups, every))[:groups]
    banned = [s for split in ("dev", "devtest") for s in flores.sentences(code, split) if len(s) >= 30]
    docs, dropped = [], 0
    for g in picks:
        t = pf.read_row_group(g, columns=["text", "language_score"]).to_pydict()
        for text, score in zip(t["text"], t["language_score"]):
            if score < min_score:
                continue
            if any(b in text for b in banned):
                dropped += 1
                continue
            docs.append(text)
    return docs, {"row_groups": picks, "documents": len(docs), "dropped_flores_overlap": dropped,
                  "bytes": sum(len(d.encode()) for d in docs)}


def path(code, name):
    return os.path.join(CACHE, "corpus", code, name)


def fetch(code, groups=60, every=20, holdout_groups=4):
    os.makedirs(os.path.dirname(path(code, "x")), exist_ok=True)
    info = {}
    for name, n, hold in (("train", groups, False), ("heldout", holdout_groups, True)):
        docs, meta = sample(code, n, every, holdout=hold)
        with open(path(code, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
        info[name] = meta
    json.dump(info, open(path(code, "info.json"), "w"), indent=1)
    return info


def documents(code, name="train"):
    with open(path(code, f"{name}.jsonl"), encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]
