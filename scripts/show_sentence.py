"""Draws one FLORES sentence as each tokenizer splits it (results/m0/sentence.html, and a PNG
when a Chromium is available: CHROME=/path/to/chrome).

    python scripts/show_sentence.py [index]
"""
import html
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lipi import flores, show, tokenizers  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "m0")
ROWS = [("eng_Latn", "o200k"), ("ory_Orya", "o200k"), ("ory_Orya", "llama4"), ("ory_Orya", "tekken"),
        ("ory_Orya", "gemma3"), ("ory_Orya", "sarvam1")]
CAPTION = {"o200k": "GPT-4o", "llama4": "Llama 4", "tekken": "Mistral NeMo<br>Krutrim-2 · Sarvam-M",
           "gemma3": "Gemma 3", "sarvam1": "Sarvam-1"}

CSS = """
body{margin:0;background:#fcfcfb;font-family:'Noto Sans','Noto Sans Oriya',sans-serif;color:#0b0b0b}
.wrap{width:1400px;padding:36px 44px}
h1{font-size:28px;margin:0 0 6px}
.sub{color:#52514e;font-size:16px;margin:0 0 26px}
.row{display:grid;grid-template-columns:230px 1fr;gap:18px;padding:14px 0;border-top:1px solid #e4e3df}
.who b{display:block;font-size:17px}.who span{color:#52514e;font-size:14px}
.n{font-size:30px;font-weight:700;color:#0b0b0b;margin-top:6px;display:block}
.n small{font-size:14px;color:#52514e;font-weight:400}
.toks{display:flex;flex-wrap:wrap;gap:3px;align-items:flex-start;font-family:'Noto Sans Oriya','Noto Sans',sans-serif;font-size:21px;line-height:1.25}
.t{padding:3px 5px;border-radius:4px;white-space:pre;position:relative}
.a{background:#cde2fb}.b{background:#e8f1fd}
.x{background:#fde2dd;outline:1.5px solid #d4553f}
.x sup{position:absolute;top:-9px;right:-5px;background:#d4553f;color:#fff;font:700 11px/1 'Noto Sans';padding:2px 4px;border-radius:7px}
.key{margin-top:18px;color:#52514e;font-size:14px}
.key i{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:-2px;margin:0 6px 0 14px}
"""


def row(code, key, idx):
    text = flores.sentences(code)[idx]
    tok = tokenizers.load(key)
    ids = tok.encode([text])[0]
    segs = show.segments(text, tok.pieces(ids))
    chips = []
    for i, (chunk, ntok, nlet) in enumerate(segs):
        label = html.escape(chunk.replace(" ", " "))
        if ntok > max(nlet, 1):
            chips.append(f'<span class="t x">{label}<sup>{ntok}</sup></span>')
        else:
            chips.append(f'<span class="t {"ab"[i % 2]}">{label}</span>')
    lang = "English" if code == "eng_Latn" else "Odia"
    return (f'<div class="row"><div class="who"><b>{lang}</b><span>{CAPTION[key]}</span>'
            f'<span class="n">{len(ids)} <small>tokens</small></span></div>'
            f'<div class="toks">{"".join(chips)}</div></div>')


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    body = "".join(row(c, k, idx) for c, k in ROWS)
    page = (f"<!doctype html><meta charset=utf-8><style>{CSS}</style><div class=wrap>"
            f"<h1>One sentence, as the models see it</h1>"
            f"<p class=sub>FLORES-200 devtest sentence {idx + 1}, the same meaning in English and Odia. "
            f"Each chip is one or more whole tokens.</p>{body}"
            f"<p class=key><i style='background:#cde2fb'></i>whole tokens"
            f"<i style='background:#fde2dd;outline:1.5px solid #d4553f'></i>"
            f"a piece the model only sees in fragments (the badge counts its tokens)</p></div>")
    path = os.path.join(OUT, "sentence.html")
    open(path, "w", encoding="utf-8").write(page)
    chrome = os.environ.get("CHROME")
    if chrome:
        subprocess.run([chrome, "--headless=new", "--no-sandbox", "--hide-scrollbars", "--force-device-scale-factor=1",
                        "--window-size=1488,810", f"--screenshot={os.path.join(OUT, 'sentence.png')}",
                        "file://" + os.path.abspath(path)], check=True, capture_output=True)


if __name__ == "__main__":
    main()
