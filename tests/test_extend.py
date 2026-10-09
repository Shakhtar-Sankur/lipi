import json

import pytest
from tokenizers import Tokenizer

from lipi import extend, tokenizers

QWEN = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"


def tok(pattern, merges=(), vocab_extra=()):
    """A tiny byte-level BPE with Qwen's pre-tokenizer layout."""
    alphabet = sorted(set(tokenizers._BYTE_DEC))
    vocab = {c: i for i, c in enumerate(alphabet)}
    cfg = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": [], "normalizer": {"type": "NFC"},
           "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
               {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": False},
               {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}]},
           "post_processor": None, "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
           "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "", "end_of_word_suffix": "",
                     "fuse_unk": False, "byte_fallback": False, "ignore_merges": False, "vocab": vocab, "merges": [list(m) for m in merges]}}
    return cfg


def pieces(cfg, text):
    t = Tokenizer.from_str(json.dumps(cfg))
    return [p for p, _ in t.pre_tokenizer.pre_tokenize_str(text)]


def test_marks_fix_keeps_words_whole_and_leaves_english_alone():
    base, fixed = tok(QWEN), extend.with_marks(tok(QWEN))
    assert len(pieces(base, "କହିଛନ୍ତି")) == 4
    assert len(pieces(fixed, "କହିଛନ୍ତି")) == 1
    english = "We now have 4-month-old mice that are non-diabetic, he added. It's 3:45 p.m.\n\n  done"
    assert pieces(base, english) == pieces(fixed, english)


def test_fix_marks_rejects_an_unknown_pattern():
    with pytest.raises(ValueError):
        extend.fix_marks("\\w+")


def test_learn_apply_and_confine():
    cfg = extend.with_marks(tok(QWEN))
    words = extend.words(Tokenizer.from_str(json.dumps(cfg)), ["ମଧୁମେହ ମଧୁମେହ ମଧୁମେହ नमस्ते नमस्ते"] * 5, extend.odia)
    assert all(any(extend.odia(c) for c in bytes(tokenizers._BYTE_DEC[x] for x in w).decode()) for w in words)
    merges = extend.learn(cfg, words, 50)
    kept = extend.confine(merges, cfg["model"]["vocab"], extend.odia_bytes)
    assert kept and all(extend.odia_bytes(a + b) for a, b in kept)
    ext, added = extend.apply(cfg, kept)
    t, t0 = Tokenizer.from_str(json.dumps(ext)), Tokenizer.from_str(json.dumps(cfg))
    odia, hindi = "ମଧୁମେହ", "नमस्ते and English"
    assert len(t.encode(odia).ids) == 1                          # the whole word is now one token
    assert t.encode(hindi).ids == t0.encode(hindi).ids           # other scripts: same ids
    assert t.decode(t.encode(odia).ids) == odia
    assert min(t.token_to_id(a) for a in added) == len(cfg["model"]["vocab"])


def test_confine_drops_merges_that_other_scripts_could_use():
    vocab = {c: i for i, c in enumerate(sorted(set(tokenizers._BYTE_DEC)))}
    lead = "à"                                                    # byte E0: starts every Indian script's characters
    odia_second = next(c for c, b in tokenizers._BYTE_DEC.items() if b == 0xAC)
    merges = [("Ġ", lead), (lead, odia_second), ("Ġ", lead + odia_second)]
    kept = extend.confine(merges, vocab, extend.odia_bytes)
    assert kept == [(lead, odia_second), ("Ġ", lead + odia_second)]


def test_indic_marks_fix_leaves_thai_alone():
    base, fixed = tok(QWEN), extend.with_marks(tok(QWEN), extend.INDIC_MARKS)
    assert len(pieces(fixed, "କହିଛନ୍ତି")) == 1
    assert len(pieces(fixed, "नमस्ते ᱥᱟᱱᱛᱟᱲᱤ")) == 2
    for text in ("น้ำตาล อย่างไร", "مَدْرَسَة", "We now have 4-month-old mice. It's 3:45 p.m."):
        assert pieces(base, text) == pieces(fixed, text)
    assert len(pieces(extend.with_marks(tok(QWEN)), "น้ำตาล")) == 1      # the unscoped fix would join it


def test_indic_bytes_marks_every_indian_script_and_nothing_else():
    def bl(text):
        enc = {b: c for c, b in tokenizers._BYTE_DEC.items()}
        return "".join(enc[b] for b in text.encode())
    for word in ("नमस्ते", "বাংলা", "ਪੰਜਾਬੀ", "ગુજરાતી", "ଓଡ଼ିଆ", "தமிழ்", "తెలుగు", "ಕನ್ನಡ", "മലയാളം", "සිංහල", "ᱥᱟᱱᱛᱟᱲᱤ", "ꯃꯤꯇꯩ", "اردو"):
        assert extend.indic_bytes(bl(word)) and all(extend.indic(c) for c in word if c.isalpha())
    for word in ("hello", "ไทย", "русский", "中文", "日本語", "한국어", "עברית", "é"):
        assert not extend.indic_bytes(bl(word)) and not any(extend.indic(c) for c in word)


def test_spread_is_a_permutation_whose_prefixes_cover_the_range():
    from lipi.corpus import _spread
    for n in (1, 2, 3, 7, 16, 100):
        assert sorted(_spread(n)) == list(range(n))
    order = _spread(100)
    assert order[0] == 50 and max(order[:4]) >= 75 and min(order[:4]) <= 25
