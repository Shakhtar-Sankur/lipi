"""Extending a byte-level BPE tokenizer (Qwen 2.5's) for a language it handles badly.

Two changes, each measurable on its own:

1. **Marks.** The pre-tokenizer's regular expression treats combining marks (`\\p{M}`: every
   Indian vowel sign and virama) as non-letters and cuts words at them. `fix_marks` lets a
   word run over its marks, as GPT-4o's expression does. Text with no combining marks
   (English, code) is cut exactly as before.

2. **Vocabulary.** New merges are learned on top of the base tokenizer's own segmentation:
   each word of the training text is first split by the base merges, and BPE then continues
   from those pieces. The new merges are appended after the base ones, so encoding replays
   the base merges first and the new ones after, exactly as in training, and text the new
   merges never match (anything outside the language trained on) gets the same ids as
   before. Each new token is a concatenation of existing ones, so its embedding can start as
   the mean of theirs.

The BPE continuation itself runs in the `tokenizers` library (Rust): each base token is
written as one private-use character, so the trainer sees the base pieces as its alphabet.
"""
import collections
import copy
import json
import re

from tokenizers import Tokenizer, models, trainers

from .tokenizers import _BYTE_DEC as BYTE_DEC

PUA = 0xF0000  # Supplementary Private Use Area-A: one character per base token


# Combining marks of the Indian scripts only (Devanagari to Sinhala, Ol Chiki, Meetei Mayek).
# Qwen's vocabulary already holds about 1,700 tokens that join letters and marks of other
# scripts (Thai, for one); its own pre-tokenizer never produces them, so their embeddings were
# never trained. Allowing every mark (`\\p{M}`) would route Thai text into those rows.
INDIC_MARKS = r"[\p{M}&&[\x{0900}-\x{0DFF}\x{1C50}-\x{1C7F}\x{ABC0}-\x{ABFF}]]"


def fix_marks(pattern, marks="\\p{M}"):
    """GPT-4-style pre-tokenizer pattern with combining marks (`marks`, a character class)
    allowed inside words."""
    out = pattern.replace("\\p{L}\\p{N}", "\\p{L}" + marks + "\\p{N}").replace("]?\\p{L}+", "]?[\\p{L}" + marks + "]+")
    if out == pattern:
        raise ValueError("pattern not recognised")
    return out


def with_marks(cfg, marks="\\p{M}"):
    """A copy of a tokenizer.json whose Split pre-tokenizer keeps marks inside words."""
    cfg = copy.deepcopy(cfg)
    for p in cfg["pre_tokenizer"].get("pretokenizers", [cfg["pre_tokenizer"]]):
        if p["type"] == "Split":
            p["pattern"]["Regex"] = fix_marks(p["pattern"]["Regex"], marks)
            return cfg
    raise ValueError("no Split pre-tokenizer")


def words(tok, docs, keep):
    """Counts of pre-tokenized pieces (in byte-level form) that contain a character `keep` accepts."""
    counts = collections.Counter()
    for doc in docs:
        if tok.normalizer is not None:
            doc = tok.normalizer.normalize_str(doc)
        for piece, (a, b) in tok.pre_tokenizer.pre_tokenize_str(doc):
            if any(keep(ch) for ch in doc[a:b]):
                counts[piece] += 1
    return counts


def learn(cfg, counts, n_new, min_frequency=2):
    """Up to `n_new` merges continuing `cfg`'s BPE on the given word counts: [(left, right)]
    in the base tokenizer's byte-level strings, in the order learned."""
    base = Tokenizer.from_str(json.dumps(cfg))
    alphabet, encoded = {}, []
    for word, n in counts.items():
        pieces = [t.value for t in base.model.tokenize(word)]
        chars = []
        for p in pieces:
            if p not in alphabet:
                alphabet[p] = chr(PUA + len(alphabet))
            chars.append(alphabet[p])
        encoded.append(("".join(chars), n))
    back = {c: p for p, c in alphabet.items()}
    t = Tokenizer(models.BPE())
    trainer = trainers.BpeTrainer(vocab_size=len(alphabet) + n_new, initial_alphabet=list(alphabet.values()),
                                  limit_alphabet=len(alphabet), min_frequency=min_frequency, show_progress=False)
    t.train_from_iterator((w for w, n in encoded for _ in range(n)), trainer=trainer, length=sum(n for _, n in encoded))
    merges = json.loads(t.to_str())["model"]["merges"]
    out = []
    for m in merges:
        a, b = m if isinstance(m, list) else m.split(" ")
        out.append(("".join(back[c] for c in a), "".join(back[c] for c in b)))
    return out


def confine(merges, vocab, marker):
    """The merges whose new token can only occur in text containing `marker` (bytes), in order.
    A merge applies only where the text holds its two halves side by side, so if every new
    token contains the marker, text without it is encoded exactly as before. Merges building
    on a dropped token are dropped too."""
    made, out = set(), []
    for a, b in merges:
        if (a in vocab or a in made) and (b in vocab or b in made) and marker(a + b):
            out.append((a, b))
            made.add(a + b)
    return out


def odia_bytes(token):
    """Whether a byte-level token string contains an Odia character's first two bytes (E0 AC or
    E0 AD: U+0B00-U+0B7F), which no other script's UTF-8 does."""
    raw = bytes(BYTE_DEC[c] for c in token)
    return b"\xe0\xac" in raw or b"\xe0\xad" in raw


def apply(cfg, merges):
    """A copy of `cfg` with `merges` appended and their new tokens added to the vocabulary,
    numbered after every existing id (added special tokens included)."""
    cfg = copy.deepcopy(cfg)
    vocab, model_merges = cfg["model"]["vocab"], cfg["model"]["merges"]
    next_id = 1 + max(max(vocab.values()), max((t["id"] for t in cfg.get("added_tokens", [])), default=-1))
    seen = {tuple(m) if isinstance(m, list) else tuple(m.split(" ")) for m in model_merges}
    added = []
    for a, b in merges:
        if (a, b) in seen:
            continue
        seen.add((a, b))
        if a + b not in vocab:
            vocab[a + b] = next_id
            added.append(a + b)
            next_id += 1
        model_merges.append([a, b] if model_merges and isinstance(model_merges[0], list) else f"{a} {b}")
    return cfg, added


def odia(ch):
    return "଀" <= ch <= "୿"


_INDIC_LEADS = re.compile(rb"\xe0[\xa4-\xb7]|\xe1\xb1|\xea\xaf|[\xd8-\xdb]")


def indic(ch):
    """A character of a script India writes its languages in: the Brahmic block U+0900-U+0DFF
    (Devanagari to Malayalam), Ol Chiki, Meetei Mayek, and Arabic (Urdu, Kashmiri, Sindhi)."""
    o = ord(ch)
    return 0x0900 <= o <= 0x0DFF or 0x1C50 <= o <= 0x1C7F or 0xABC0 <= o <= 0xABFF or 0x0600 <= o <= 0x06FF


def indic_bytes(token):
    """Whether a byte-level token contains the start of a character in those scripts. Their
    lead bytes (E0 A4-B7, E1 B1, EA AF, D8-DB) occur in no other script's UTF-8 except Arabic's,
    which shares Urdu's block: Arabic and Persian text can be affected, nothing else."""
    return bool(_INDIC_LEADS.search(bytes(BYTE_DEC[c] for c in token)))
