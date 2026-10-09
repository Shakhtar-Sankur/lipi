"""What a tokenizer does to a text, measured on parallel sentences.

For each (tokenizer, language) over the same sentences:

- tokens, and the *token tax*: tokens in the language / tokens for the same sentences in
  English. Price, generation time and the share of a context window all scale with it;
- tokens per letter: a letter here is an extended grapheme cluster (Unicode `\\X`, which joins
  a consonant cluster such as ସ୍ତ୍ରୀ into one, as a reader sees it);
- broken letters: the share of letters that a token boundary cuts in two, so the model never
  sees them whole;
- byte fragments: the share of tokens that are not even whole characters (a piece of a
  character's UTF-8 encoding);
- unknown tokens: text the tokenizer cannot represent at all (<unk>).

The byte measures need each token's bytes to concatenate back to the text exactly, allowing for
a leading space (SentencePiece) and Unicode normalisation (Qwen applies NFC, NLLB NFKC-style
rules); sentences where they still do not (<unk>, other rewriting) are counted in `exact` and
left out of those measures only.
"""
import unicodedata

import regex

_LETTER = regex.compile(r"\X")


def letters(text):
    return _LETTER.findall(text)


def sentence(tok, text, ids):
    """Counts for one sentence; byte measures only if the pieces round-trip to the text."""
    r = {"tokens": len(ids), "letters": len(letters(text)), "chars": len(text),
         "bytes": len(text.encode()), "words": len(text.split()),
         "unk": sum(1 for i in ids if tok.is_unk(i))}
    pieces = tok.pieces(ids)
    if pieces is None or any(p is None for p in pieces):
        return r
    joined = b"".join(pieces)
    for form in (None, "NFC", "NFKC"):   # some tokenizers normalise before splitting
        t = text if form is None else unicodedata.normalize(form, text)
        raw = t.encode()
        if joined in (raw, b" " + raw):
            shift = 0 if joined == raw else 1
            text = t
            break
    else:
        return r
    # token boundaries, as byte offsets into the text
    cuts, pos = set(), -shift
    for p in pieces[:-1]:
        pos += len(p)
        cuts.add(pos)
    fragments = 0
    for p in pieces:
        try:
            p.decode("utf-8")
        except UnicodeDecodeError:
            fragments += 1
    broken, off = 0, 0
    for g in letters(text):
        n = len(g.encode())
        if any(off < c < off + n for c in cuts):
            broken += 1
        off += n
    r.update(exact=1, exact_tokens=len(ids), exact_letters=len(letters(text)), fragments=fragments, broken=broken)
    return r


def corpus(tok, texts):
    """Sums of the per-sentence counts over a list of sentences."""
    total = {}
    for text, ids in zip(texts, tok.encode(texts)):
        for k, v in sentence(tok, text, ids).items():
            total[k] = total.get(k, 0) + v
    total["sentences"] = len(texts)
    total.setdefault("exact", 0)
    return total
