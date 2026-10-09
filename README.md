# lipi

**The token tax on Indian languages, measured and then fixed.** A language model reads and
writes in tokens, and it is priced, timed and limited in tokens. The same sentence costs far
more tokens in an Indian language than in English, so Indian users pay more per answer, wait
longer for it, and fit less of their text in the model's memory. lipi measures that tax for
every Indian language in FLORES-200 across 15 tokenizers that ship with today's models, and
then works to remove it from an open model (the plan is below).

*Lipi* (ଲିପି, लिपि, লিপি) means "script, writing" in Odia, Hindi and Bengali.

![One Odia sentence as five tokenizers split it](results/m0/sentence.png)

## What M0 found

The same 1,012 sentences (FLORES-200 devtest, professionally translated), counted by each
tokenizer and divided by the count for English (`results/m0/token_tax.csv`):

![The token tax across 24 Indian languages and 13 distinct tokenizers](results/m0/token_tax_heatmap.png)

- **Odia pays the most of any major Indian language, on almost every model.** On GPT-4o the
  same text costs **5.0×** the tokens of English and **3.2×** the tokens of Hindi (1.57×). On
  Llama 4 it is **8.4×** (Hindi: 1.65×), on Llama 3 **12.2×**, on Qwen 3 **9.7×**. GPT-4o cuts
  **53%** of Odia letters into pieces; on Llama 3, **99%** of the tokens it produces for Odia
  are not even whole characters but fragments of their bytes.
- **Two Indian models inherit the worst Odia tax measured.** Krutrim-2 (Ola) and Sarvam-M
  (Sarvam AI) ship Mistral's Tekken tokenizer: Krutrim-2's tokenizer file is byte-identical to
  Mistral NeMo's, and Sarvam-M's gives the same token counts in every language. Odia costs **12.9×**
  English there, **7.2×** Hindi; a single letter (ର୍ତ୍ତ) takes 15 tokens, one per byte.
- **There are two different causes, and some tokenizers have both.** Before a tokenizer uses
  its vocabulary, a regular expression cuts the text into pieces, and no token can span two
  pieces. The expression in GPT-4's, Llama 3's and Qwen's tokenizers matches letters
  (`\p{L}`) but not combining marks (`\p{M}`), and every Indian vowel sign and virama is a
  combining mark: କହିଛନ୍ତି ("has said") is cut into କହ · ିଛନ · ୍ତ · ି before the vocabulary
  is consulted. However large their vocabulary, these tokenizers cannot write Odia in fewer
  than **2.33×** the tokens of English, Hindi in fewer than 2.37×, Tamil in fewer than
  2.66×. GPT-4o, Llama 4, DeepSeek-V3 and Mistral's Tekken match marks as part of words
  (their floor for Odia is 0.84–0.85× English), so their tax is the other cause: a vocabulary
  with few Odia tokens. Tekken shows it in pure form: a floor of 0.84× and a tax of 12.9×.
- **Santali, in its own Ol Chiki script, is the most expensive language measured**: 13.7× on
  GPT-4o (worse than GPT-4's 12.7×), 12.4× on Mistral, 11.8× even on Sarvam-1. 7.4 million
  people speak it.
- **Weighted by speakers** (Census 2011, 1.16 billion people across the languages measured),
  the average Indian's text costs **1.9×** English on GPT-4o, **2.4×** on Llama 4, **4.8×** on
  Llama 3 and **5.3×** on Qwen 3. On Llama 3 and Qwen 3 every language measured costs at
  least 2×.
- **It is fixable, and the fix is known to work for tokenization.** Tokenizers built with
  Indian languages in mind cost 1.1–1.5× English for the major languages: Sarvam-1, IndicBERT
  v2 (AI4Bharat), BLOOM, NLLB-200. Gemma 3, with a 262K vocabulary, is the best of the
  general-purpose models (1.2–2.7×, except Odia at 3.5× and Santali at 5.3×). What a better tokenizer does to a model's
  speed and quality is what M1 and M2 measure.
- **A tokenizer's own blind spots show.** Sarvam-1, the best tokenizer here for Odia (1.46×),
  costs 7.4× for Urdu and 6.8× for Sindhi, scripts outside the ten languages it was built for.

![GPT-4o: the token tax by language](results/m0/tax_o200k.png)

### How it is measured

- **Text**: FLORES-200 devtest (Meta, CC BY-SA 4.0), the same 1,012 sentences in English and
  every Indian language it has: 19 of the 22 languages of the Eighth Schedule (Bodo, Dogri and
  Konkani are not in FLORES-200; IN22, which has all 22, needs a Hugging Face login and is
  next), Kashmiri in both its scripts, and Awadhi, Bhojpuri, Chhattisgarhi and Magahi.
- **Tokenizers** (`lipi/tokenizers.py`), each pinned to a commit: GPT-4o (`o200k_base`, also
  gpt-oss), GPT-4 (`cl100k_base`, also Phi-4), Llama 3, Llama 4, Gemma 3, Qwen 3, DeepSeek-V3,
  Mistral 7B v0.3, Mistral NeMo (Tekken), Krutrim-2, Sarvam-M, Sarvam-1, BLOOM, NLLB-200 and
  IndicBERT v2. Llama, Gemma and Mistral NeMo come from ungated re-uploads of the official
  files. That Phi-4 and gpt-oss tokenize like `cl100k_base` and `o200k_base` was checked on
  50 Odia sentences each.
- **Token tax**: tokens for the 1,012 sentences in a language / tokens for the same sentences
  in English, with no special tokens. Price, generation time and the share of a context
  window all scale with it.
- **Letters**: extended grapheme clusters (Unicode `\X`), which keep a conjunct such as ସ୍ତ୍ରୀ
  together as one letter, the way a reader sees it.
- **Floor**: the number of pieces the pre-tokenizer's regular expression cuts the text into
  (`Tokenizer.floor`), the fewest tokens any vocabulary could reach; reported for the
  byte-level and tiktoken tokenizers, whose pre-tokenizers do the cutting.
- **Broken letters / byte fragments**: from the exact bytes of every token, checked to
  concatenate back to the sentence (allowing for SentencePiece's leading space and the
  Unicode normalisation Qwen and NLLB apply). Every tokenizer passes on every sentence except
  NLLB-200, whose `<unk>` tokens and text rewriting leave 54–99% of sentences checkable; its
  byte measures cover those only. IndicBERT v2 (WordPiece) has no byte mapping; only its
  counts are used.
- **Speakers**: Census of India 2011, scheduled languages (Kashmiri counted once).

### Limits

- Token counts are not the whole cost: a tokenizer that splits Odia into bytes may also make
  the model worse at Odia, which M0 does not measure; M1 does.
- FLORES-200 is formal, translated news and travel text. Conversation, code-mixed Hinglish or
  romanised Indian languages will tax differently.
- The tax is relative to English on these sentences. Prices per token differ between providers;
  the multiple applies to whatever the price is.
- Measuring tokenizer unfairness is not new (Petrov et al., *Language Model Tokenizers
  Introduce Unfairness Between Languages*, NeurIPS 2023; Ahia et al., *Do All Languages Cost
  the Same?*, EMNLP 2023). lipi's part is current models, every Indian language FLORES-200
  has, letters that tokenizers break, and then the fix with its effect on speed and quality.

## M1: the fix

### Tokenizer

Both causes, removed from Qwen 2.5's tokenizer (`lipi/extend.py`, `scripts/extend_tokenizer.py`):

1. **Marks.** The pre-tokenizer's expression lets a word run over its combining marks, as
   GPT-4o's does. Text without combining marks (English, code) is cut exactly as before.
2. **Vocabulary.** New merges are learned on 85 MB of Odia web text (FineWeb-2, 20,000
   documents; none contain a FLORES sentence) by continuing BPE from Qwen's own segmentation,
   and appended after Qwen's 151,387 merges. Every new token contains the first two bytes
   of an Odia character (E0 AC or E0 AD), which no other script's UTF-8 does, so the new
   merges cannot fire on any other text: all 24 other languages keep exactly the same ids.

On FLORES-200 devtest (held out), tokens as a multiple of English (`results/m1/tokenizer.json`):

| Qwen 2.5 tokenizer | new tokens | Odia | Hindi | English |
|---|---|---|---|---|
| as shipped | 0 | 9.70× | 4.42× | 1× |
| marks fix only | 0 | 9.70× | 4.42× | 1× (same tokens) |
| new Odia tokens only | 1K / 4K / 16K | 2.53× / 2.38× / **2.34×** | 4.42× (same ids) | 1× (same ids) |
| **both** | 1K / 4K / **16K** / 32K | 1.98× / 1.44× / **1.13×** / 1.04× | 4.42× | 1× (same tokens) |

New tokens alone stop at the pre-tokenizer's floor (2.33×, from M0); the marks fix alone does
nothing, because the vocabulary has no Odia words to use it with. Together they take Odia
from 9.70× English to **1.13×** with 16K new tokens (8.6× fewer tokens for the same text) and
1.04× with 32K. The marks fix changes the ids of 18 other languages (their words are no
longer cut at vowel signs) and lowers their token counts by at most 0.5% (Kashmiri in Perso-Arabic script: 0.46% fewer).

The 16K tokenizer is `results/m1/tokenizers/qwen25-odia-marks-vocab-16k.json`.

### Model

Fewer tokens help only if the model still understands the text. Qwen 2.5 0.5B gets the
extended vocabulary (each new token's embedding starts as the mean of the pieces it
replaces; `lipi/model.py`) and continued pre-training on Odia web text plus a quarter as much
English, so English is not forgotten. Three arms, everything else identical
(`scripts/train_m1.py`, `scripts/kaggle_m1.sh`; Kaggle's 2× T4, DDP with ZeRO-1, one run per
arm, `results/m1/kaggle-full-2026-10-09.txt`):

- **A**: Qwen's own tokenizer, on 24 MB of Odia and 6 MB of English: 19.9M tokens;
- **B**: the extended tokenizer, on the same 30 MB of text: 3.5M tokens;
- **C**: the extended tokenizer, for as many tokens as A (the same GPU time): 135 MB of Odia
  and 34 MB of English.

Only the new tokens' embedding rows learn at a high rate (5e-4); everything else, the
original embedding rows (which are also the output layer) included, at 5e-5. Measured on
text no arm trained on:

| | Qwen 2.5 0.5B | extended, untrained | A | B | C |
|---|---|---|---|---|---|
| tokens for the 1,012 Odia FLORES sentences | 267,893 | 31,211 | 267,893 | **31,211** | **31,211** |
| training time on 2× T4 | | | 104 min | **19 min** | 108 min |
| Odia bits per byte, FLORES devtest | 1.444 | 1.183 | 0.751 | 0.782 | **0.674** |
| Odia bits per byte, 500 held-out web documents | 1.130 | 1.103 | 0.575 | 0.608 | **0.461** |
| English bits per byte, FLORES devtest | **1.076** | 1.205 | 1.164 | 1.106 | 1.127 |
| Odia → English translation, chrF++ (200 sentences) | 16.0 | 11.3 | 0.0 | 15.4 | **24.8** |
| English → Odia translation, chrF++ (200 sentences) | 8.7 | 2.6 | **10.8** | 3.7 | 8.3 |
| Belebele reading comprehension, Odia (900 questions) | 30.6% | 23.0% | 29.2% | 31.3% | 30.1% |
| time to score Belebele in Odia | 382 s | 46 s | 385 s | **46 s** | **44 s** |

![Odia quality and training time](results/m1/m1.png)

What the numbers show:

- **The same text, 5.4× cheaper to train on.** B reads exactly A's 30 MB in 19 minutes
  instead of 104, and ends within 4% of A's Odia bits per byte on FLORES (6% on web text),
  while forgetting less English (+2.8% bits per byte against A's +8.2%).
- **The same GPU time, a better model.** In A's 104 minutes the extended tokenizer reads
  5.6× as much Odia, and C beats A by 10% on FLORES and 20% on held-out web text. C also
  translates Odia into English better than the untrained model (chrF++ 24.8 against 16.0),
  while A's translations share almost nothing with the references (0.03). This run kept only
  five outputs per model, so why A fails is not yet checked; evaluations now keep them all.
- **Inference is 8.4× cheaper per Odia question.** Scoring the 900 Belebele questions in Odia
  takes 46 s instead of 382 s, because each question is 8.6× fewer tokens.
- **Even untrained, the extended model predicts Odia better** (1.18 bits per byte against
  1.44): starting each new token at the mean of its pieces already works.

What they do not show, or show against the fix:

- **Belebele does not move.** Every model scores 23–31% in Odia (chance is 25%; one standard
  error is 1.5 points). A 0.5B model cannot answer reading-comprehension questions in Odia
  with or without the fix, so this benchmark cannot tell the arms apart.
- **English → Odia translation got worse with the extended tokenizer** (8.3 and 3.7 against
  A's 10.8): with five examples in the prompt, B's and C's sample outputs copy the English
  sentence back instead of translating it. A small base model translating into a language it barely
  knows is fragile in every arm (the untrained model copies an example sentence instead);
  this needs instruction tuning, not a tokenizer, and is left as it is.
- **English is still worse than before training** in every arm (+2.8% to +8.2% bits per
  byte); 20% English in the mix limits the damage but does not prevent it.
- One run per arm, so differences of a few percent are within what a second seed could move.

## Plan

| | Milestone | State |
|---|---|---|
| M0 | Measure the token tax: 24 Indian languages × 15 tokenizers; tokens, letters broken, byte fragments, the pre-tokenizer's floor; charts | done |
| M1 | Fix it in an open model: train Indic tokens, extend a small model's vocabulary (Qwen 2.5), initialise and train the new embeddings on Kaggle's 2× T4; measure tokens saved and quality (bits per byte, translation, comprehension) before and after | done (one run per arm) |
| M2 | Speed and cost end to end: time to answer and tokens per second per language before and after, on a T4; speculative decoding; cost per 1,000 answers | |
| M3 | An interactive token-tax page (type a sentence, see each model's tokens and cost), all 22 scheduled languages with IN22, write-up | |

## Run it

```
pip install tokenizers tiktoken regex matplotlib pytest
python scripts/token_tax.py          # all tokenizers x 25 languages (about 2 minutes on 4 CPUs)
python scripts/charts.py             # the charts above
CHROME=/path/to/chrome python scripts/show_sentence.py 0    # the sentence picture
python -m pytest tests               # LIPI_ALL_TOKENIZERS=1 also round-trips every tokenizer
```

FLORES-200 and the tokenizers download on first use into `~/.cache/lipi` (`LIPI_CACHE`).
