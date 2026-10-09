"""Where a tokenizer cuts a sentence, in units a reader can see.

A token can be smaller than a letter (a few bytes of one character), so tokens cannot always
be drawn on their own. `segments` groups the text into the smallest pieces whose edges are
both token edges and letter edges: each holds whole tokens and whole letters, and says how
many of each. A segment of one letter and four tokens is a letter the model only ever sees
in four fragments.
"""
from . import measure


def segments(text, pieces):
    """[(text, tokens, letters)] covering `text`, or None if the pieces do not round-trip."""
    raw = text.encode()
    joined = b"".join(pieces)
    shift = 1 if joined == b" " + raw else 0
    if joined[shift:] != raw:
        return None
    cuts, pos = set(), -shift
    for p in pieces[:-1]:
        pos += len(p)
        cuts.add(pos)
    edges, off = {0}, 0
    for g in measure.letters(text):
        off += len(g.encode())
        edges.add(off)
    keep = sorted((cuts & edges) | {0, len(raw)})
    out = []
    for a, b in zip(keep, keep[1:]):
        chunk = raw[a:b].decode()
        tokens = 1 + sum(1 for c in cuts if a < c < b)
        out.append((chunk, tokens, len(measure.letters(chunk))))
    return out
