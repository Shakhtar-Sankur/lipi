"""The model half: a model whose tokenizer was extended, and how to compare models fairly.

`extend_embeddings` grows the embedding matrix to the extended vocabulary and starts each new
token's row at the mean of the rows of the base tokens it replaces (its bytes, as the base
tokenizer splits them): "fast vocabulary transfer" (Gee et al., 2022). With tied embeddings
(Qwen 2.5 0.5B) the same row is the new token's output vector.

Models that tokenize differently are compared in bits per byte (`evaluate.bits_per_byte`):
per-token loss is not comparable across tokenizers, since fewer, longer tokens each carry
more information.
"""
import json

import torch
from tokenizers import Tokenizer


def extend_embeddings(model, base_cfg, ext_cfg, multiple=128):
    """Resize `model` to the extended vocabulary and initialise the new rows. Returns the new
    token ids and, for each, the base ids it was initialised from."""
    vocab = ext_cfg["model"]["vocab"]
    old = base_cfg["model"]["vocab"]
    new = sorted((i, t) for t, i in vocab.items() if t not in old)
    size = max(max(vocab.values()) + 1, model.get_input_embeddings().weight.shape[0])
    size = -(-size // multiple) * multiple
    model.resize_token_embeddings(size, mean_resizing=False)
    emb = model.get_input_embeddings().weight
    base = Tokenizer.from_str(json.dumps(base_cfg))
    sources = {}
    with torch.no_grad():
        for i, t in new:
            ids = [base.token_to_id(p.value) for p in base.model.tokenize(t)]
            emb[i] = emb[ids].mean(0)
            sources[i] = ids
        if not model.config.tie_word_embeddings:
            head = model.get_output_embeddings().weight
            for i, ids in sources.items():
                head[i] = head[ids].mean(0)
    return sources
