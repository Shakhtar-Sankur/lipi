import os

import pytest

from lipi import languages, measure, show, tokenizers


class Fake:
    """A tokenizer whose tokens are given byte slices of the text."""
    style = "bytelevel"

    def __init__(self, cuts):
        self.cuts = cuts

    def encode(self, texts):
        return [list(range(len(self._pieces(t)))) for t in texts]

    def _pieces(self, text):
        raw, edges = text.encode(), [0] + self.cuts
        return [raw[a:b] for a, b in zip(edges, edges[1:] + [len(raw)])]

    def pieces(self, ids):
        return self._pieces(self.text)[:len(ids)]

    def is_unk(self, i):
        return False


def run(text, cuts):
    tok = Fake(cuts)
    tok.text = text
    return measure.sentence(tok, text, tok.encode([text])[0])


def test_a_conjunct_is_one_letter():
    assert measure.letters("ସ୍ତ୍ରୀ") == ["ସ୍ତ୍ରୀ"]           # s + virama + t + virama + r + ii
    assert len(measure.letters("ଓଡ଼ିଆ")) == 3
    assert len(measure.letters("नमस्ते")) == 3


def test_whole_letters_are_not_broken():
    r = run("ab cd", [2])                       # "ab" | " cd"
    assert (r["tokens"], r["broken"], r["fragments"], r["exact"]) == (2, 0, 0, 1)


def test_byte_by_byte_breaks_every_letter():
    text = "ଓଡ଼ିଆ"                               # 5 code points, 15 bytes, 3 letters
    r = run(text, list(range(1, len(text.encode()))))
    assert r["tokens"] == 15
    assert r["broken"] == 3                     # every letter is cut
    assert r["fragments"] == 15                 # no token is a whole character


def test_a_cut_between_characters_inside_a_letter():
    text = "ଡ଼ି"                                 # one letter: DDA + nukta + vowel sign I
    r = run(text, [3])                          # cut after DDA: whole characters, broken letter
    assert (r["broken"], r["fragments"]) == (1, 0)


def test_segments_group_fragments_into_whole_letters():
    text = "ab ଓଡ଼ିଆ"
    raw = text.encode()
    cuts = [2] + list(range(3, len(raw)))       # "ab" | " " ... then one token per byte
    pieces = [raw[a:b] for a, b in zip([0] + cuts, cuts + [len(raw)])]
    segs = show.segments(text, pieces)
    assert "".join(s for s, _, _ in segs) == text
    # "ab" is one token of two letters; then one token per byte: ଓ is 3 bytes, ଡ଼ି (DDA, nukta,
    # vowel sign) one letter of 9 bytes, ଆ 3 bytes
    assert segs == [("ab", 1, 2), (" ", 1, 1), ("ଓ", 3, 1), ("ଡ଼ି", 9, 1), ("ଆ", 3, 1)]


def test_language_table():
    codes = [l.code for l in languages.ALL]
    assert len(codes) == len(set(codes)) == 25
    assert sum(l.scheduled for l in languages.INDIC) == 20    # 19 languages, Kashmiri in two scripts
    assert languages.BY_CODE["ory_Orya"].native == "ଓଡ଼ିଆ"


SAMPLE = [  # FLORES-200 devtest, sentence 1
    "\"We now have 4-month-old mice that are non-diabetic that used to be diabetic,\" he added.",
    "ସେ ଆହୁରି ମଧ୍ୟ କହିଛନ୍ତି ଯେ, ‘‘ଆମର ବର୍ତ୍ତମାନ 4 ମାସର ମୂଷା ଅଛି, ଯିଏ ମଧୁମେହ ଆକ୍ରାନ୍ତ ନୁହେଁ, ଯିଏ ପୂର୍ବରୁ ମଧୁମେହରେ ଆକ୍ରାନ୍ତ ଥିଲା।\"",
    "उन्होंने कहा “कि अब हमारे पास 4 महीने उम्र वाले चूहे हैं जिन्हें मधुमेह नहीं है जो मधुमेह के रोगी थे। ”",
    "\u2063ᱩᱱᱤ ᱞᱟᱹᱭ ᱠᱮᱜ-ᱟᱭ, \"ᱟᱞᱮ ᱴᱷᱮᱱ ᱔ ᱪᱟᱸᱫᱚ ᱨᱤᱱᱤᱡ ᱜᱩᱰᱩ ᱢᱮᱱᱟᱭᱟ ᱡᱟᱦᱟᱸᱭ ᱫᱚ ᱵᱟᱭ ᱰᱟᱭᱵᱮᱴᱤᱥᱟ ᱩᱱᱤ ᱯᱩᱭᱞᱩ ᱰᱟᱭᱵᱮᱴᱤᱥ ᱛᱟᱦᱮᱸᱠᱟᱱᱟ᱾\"",
    "انہوں نے مزید بتایا کہ، \"اب ہمارے پاس غیر ذیابیس والے 4 مہینے کی عمر کے چوہے ہیں جنہیں شوگر ہوجایا کرتا تھا۔\"",
]


@pytest.mark.skipif(not os.environ.get("LIPI_ALL_TOKENIZERS"), reason="downloads every tokenizer; set LIPI_ALL_TOKENIZERS=1")
@pytest.mark.parametrize("key", [s.key for s in tokenizers.SPECS if s.key not in ("indicbert2", "nllb")])
def test_every_tokenizer_round_trips(key):
    tok = tokenizers.load(key)
    for text, ids in zip(SAMPLE, tok.encode(SAMPLE)):
        assert measure.sentence(tok, text, ids).get("exact") == 1, (key, text)


def test_gpt4o_round_trips_and_cuts_odia():
    tok = tokenizers.load("o200k")
    eng, odia = (measure.sentence(tok, t, i) for t, i in zip(SAMPLE[:2], tok.encode(SAMPLE[:2])))
    assert eng["exact"] == odia["exact"] == 1
    assert (eng["tokens"], odia["tokens"]) == (22, 124)     # pinned: the encoding files are hash-checked
    assert eng["broken"] == 0 and odia["broken"] > 0
