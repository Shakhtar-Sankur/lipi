"""The tokenizers measured, pinned to exact revisions, behind one interface.

Every tokenizer returns, for a text, its token ids and (where the format allows) the exact
bytes of each token, so that a token can be checked against the text it came from: whether it
holds whole characters, whether it cuts a letter (a grapheme cluster) in two.

- byte-level BPE (GPT-2 style: Llama 3/4, Qwen, DeepSeek, BLOOM, Mistral's Tekken): a token's
  string maps back to bytes through the GPT-2 byte-to-unicode table;
- SentencePiece style (Gemma, Mistral v0.3, Sarvam-1, NLLB): '▁' is a space, '<0xNN>' a raw byte;
- tiktoken (OpenAI): the library gives each token's bytes directly;
- WordPiece (IndicBERT v2): no byte mapping; only the token count is measured.
"""
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field

CACHE = os.environ.get("LIPI_CACHE", os.path.expanduser("~/.cache/lipi"))


@dataclass(frozen=True)
class Spec:
    key: str
    label: str          # shown in tables and charts
    used_by: str        # models that ship this tokenizer
    kind: str           # "hf" or "tiktoken"
    source: str         # Hugging Face repo, or tiktoken encoding name
    revision: str = ""  # commit of the Hugging Face repo
    note: str = ""


SPECS = [
    Spec("o200k", "GPT-4o", "GPT-4o, GPT-4.1, o1/o3/o4, gpt-oss", "tiktoken", "o200k_base"),
    Spec("cl100k", "GPT-4", "GPT-4, GPT-3.5, Phi-4", "tiktoken", "cl100k_base"),
    Spec("llama3", "Llama 3", "Llama 3, 3.1, 3.2, 3.3", "hf", "unsloth/Llama-3.2-1B",
         "9535bd9b1d1dea6acafbdc4813b728796aeb28da", "an ungated re-upload of meta-llama/Llama-3.2-1B"),
    Spec("llama4", "Llama 4", "Llama 4 Scout and Maverick", "hf", "unsloth/Llama-4-Scout-17B-16E-Instruct",
         "afd8e498c87bda51c7ea8ec68ea2f7c066e6340b", "an ungated re-upload of meta-llama/Llama-4-Scout-17B-16E-Instruct"),
    Spec("gemma3", "Gemma 3", "Gemma 3", "hf", "unsloth/gemma-3-1b-it",
         "5b11413a10db4e486ef16a20101fd028f8f2499c", "an ungated re-upload of google/gemma-3-1b-it"),
    Spec("qwen3", "Qwen 3", "Qwen 2.5, Qwen 3", "hf", "Qwen/Qwen3-0.6B",
         "c1899de289a04d12100db370d81485cdf75e47ca"),
    Spec("deepseek3", "DeepSeek-V3", "DeepSeek-V3, DeepSeek-R1", "hf", "deepseek-ai/DeepSeek-V3",
         "e815299b0bcbac849fa540c768ef21845365c9eb"),
    Spec("tekken", "Mistral Tekken", "Mistral NeMo", "hf", "unsloth/Mistral-Nemo-Base-2407",
         "9b378d3ad787067d4bcee251d79243f1b55b633d", "an ungated re-upload of mistralai/Mistral-Nemo-Base-2407"),
    Spec("mistral3", "Mistral v0.3", "Mistral 7B v0.3", "hf", "mistralai/Mistral-7B-v0.3",
         "caa1feb0e54d415e2df31207e5f4e273e33509b1"),
    Spec("krutrim2", "Krutrim-2", "Krutrim-2 (Ola Krutrim)", "hf", "krutrim-ai-labs/Krutrim-2-instruct",
         "b2f4411404153b4afe4b8c721bd5ca5dc3481b89"),
    Spec("sarvamm", "Sarvam-M", "Sarvam-M (Sarvam AI, from Mistral Small 3.1)", "hf", "sarvamai/sarvam-m",
         "01534a53c46f2788e392dbb3d994e0fa8f04d3fd"),
    Spec("sarvam1", "Sarvam-1", "Sarvam-1", "hf", "sarvamai/sarvam-1",
         "e9607337286ddf496d4a2562b194e489dcf3feea"),
    Spec("bloom", "BLOOM", "BLOOM, BLOOMZ", "hf", "bigscience/bloom-560m",
         "ac2ae5fab2ce3f9f40dc79b5ca9f637430d24971"),
    Spec("nllb", "NLLB-200", "NLLB-200 (translation)", "hf", "facebook/nllb-200-distilled-600M",
         "f8d333a098d19b4fd9a8b18f94170487ad3f821d"),
    Spec("indicbert2", "IndicBERT v2", "IndicBERT v2 (AI4Bharat)", "hf", "ai4bharat/IndicBERTv2-MLM-only",
         "8598f13fe52443bc3fc054fcd665944560145b5c"),
]
BY_KEY = {s.key: s for s in SPECS}


def _download(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    return path


def _gpt2_byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_TOKEN = re.compile(r"<0x([0-9A-Fa-f]{2})>")


@dataclass
class Tokenizer:
    spec: Spec
    style: str                     # "bytelevel", "spm", "tiktoken" or "wordpiece"
    vocab_size: int
    _impl: object = field(repr=False)
    _unk_id: int = None

    def encode(self, texts):
        """Token ids for each text, with no special tokens added."""
        if self.style == "tiktoken":
            return [self._impl.encode_ordinary(t) for t in texts]
        return [e.ids for e in self._impl.encode_batch(texts, add_special_tokens=False)]

    def pieces(self, ids):
        """The bytes of each token, or None where the format has no byte mapping (WordPiece)
        or the token is unknown (<unk>)."""
        if self.style == "tiktoken":
            return [self._impl.decode_single_token_bytes(i) for i in ids]
        if self.style == "wordpiece":
            return None
        out = []
        for i in ids:
            if i == self._unk_id:
                out.append(None)
                continue
            t = self._impl.id_to_token(i)
            if self.style == "bytelevel":
                out.append(bytes(_BYTE_DEC[c] for c in t))
            else:
                m = _BYTE_TOKEN.fullmatch(t)
                out.append(bytes([int(m.group(1), 16)]) if m else t.replace("▁", " ").encode())
        return out

    def is_unk(self, i):
        return self._unk_id is not None and i == self._unk_id


_BYTE_DEC = _gpt2_byte_decoder()


def load(key):
    spec = BY_KEY[key]
    if spec.kind == "tiktoken":
        import tiktoken
        enc = tiktoken.get_encoding(spec.source)
        return Tokenizer(spec, "tiktoken", enc.n_vocab, enc)
    from tokenizers import Tokenizer as HF
    path = _download(f"https://huggingface.co/{spec.source}/resolve/{spec.revision}/tokenizer.json",
                     os.path.join(CACHE, "tokenizers", key, "tokenizer.json"))
    tok = HF.from_file(path)
    cfg = json.load(open(path, encoding="utf-8"))
    model = cfg["model"]["type"]
    if model == "WordPiece":
        style = "wordpiece"
    elif "ByteLevel" in json.dumps(cfg.get("decoder")) or "ByteLevel" in json.dumps(cfg.get("pre_tokenizer")):
        style = "bytelevel"
    else:
        style = "spm"
    unk = cfg["model"].get("unk_token")
    unk_id = cfg["model"].get("unk_id")
    if unk_id is None and isinstance(unk, str):
        unk_id = tok.token_to_id(unk)
    return Tokenizer(spec, style, tok.get_vocab_size(with_added_tokens=True), tok, unk_id)
