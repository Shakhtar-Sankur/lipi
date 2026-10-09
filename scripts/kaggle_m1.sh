#!/bin/bash
# M1 on Kaggle ("GPU T4 x2", Internet on). In a notebook cell:
#   !cd /tmp && rm -rf lipi && git clone -q --depth 1 https://github.com/Shakhtar-Sankur/lipi && RUN=smoke bash lipi/scripts/kaggle_m1.sh
#
# RUN=smoke : about 15 minutes; every step on a little data, to find problems first.
# RUN=full  : the measured run (several hours: "Save Version -> Save & Run All").
#
# Models compared, all Qwen 2.5 0.5B:
#   base  as shipped, Qwen's tokenizer                        (no training)
#   init  extended tokenizer, new rows = means of their pieces (no training)
#   A     Qwen's tokenizer, trained on the Odia+English text
#   B     extended tokenizer, trained on the same text         (same data as A: fewer tokens, less compute)
#   C     extended tokenizer, trained on as many tokens as A   (same compute as A: more text)
set -eo pipefail
RUN=${RUN:-smoke}
ARMS=${ARMS:-"A B C"}
cd "$(dirname "$0")/.."
echo "== lipi M1 ($RUN, arms: $ARMS): lipi $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
NGPU=$(nvidia-smi -L | wc -l)
pip install -q sacrebleu 2>&1 | grep -v -i warn || true

python - <<'EOF'
import os, urllib.request
from lipi import corpus, flores
d = os.path.join(corpus.CACHE, "models", "qwen25-0.5b"); os.makedirs(d, exist_ok=True)
for f in ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
    if not os.path.exists(os.path.join(d, f)):
        urllib.request.urlretrieve(f"https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/060db6499f32faf8b98477b0a26969ef7d8b9987/{f}", os.path.join(d, f))
flores.root()
full = os.environ.get("RUN") == "full"
for code, g, every, hold in (("ory_Orya", 60 if full else 4, 20, 4), ("eng_Latn", 20 if full else 2, 50, 2)):
    info = corpus.path(code, "info.json")
    if not os.path.exists(info) or len(__import__("json").load(open(info))["train"]["row_groups"]) < g:
        info = corpus.fetch(code, groups=g, every=every, holdout_groups=hold)
        print(code, {k: {x: y for x, y in v.items() if x != "row_groups"} for k, v in info.items()}, flush=True)
EOF

mkdir -p runs
OUT=runs/evals.jsonl
if [ "$RUN" = smoke ]; then
  TRAIN="--odia-mb 1 --english-mb 0.25 --max-steps 20"; EVAL="--quick"
else
  TRAIN="--odia-mb 24 --english-mb 6"; EVAL=""
fi

ev() {  # ev NAME GPU ARGS...: evaluate on one GPU, in the background
  local name=$1 gpu=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu python scripts/eval_m1.py --name $name $EVAL --out $OUT "$@" > runs/eval-$name.log 2>&1 &
}

echo "== evaluate the untrained models"
ev base 0 --tokenizer qwen
ev init $((NGPU > 1 ? 1 : 0)) --tokenizer extended
wait
grep -h '"phase"' runs/eval-base.log runs/eval-init.log || true

train() {  # train NAME ARGS...
  echo "== train $1"
  local name=$1; shift
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node $NGPU scripts/train_m1.py --out runs/$name "$@" 2>&1 | tee runs/train-$name.log | \
    grep --line-buffered -E '^\{' | python -c "import sys,json
for l in sys.stdin:
    r=json.loads(l)
    if 'phase' in r or r['step'] % 100 == 0 or r.get('progress', 0) > 0.99: print(l, end='', flush=True)" \
    || { echo "== train $name failed:"; grep -v -E "torch/distributed|^\s+[~^]+$" runs/train-$name.log | grep -E -B3 -A12 "Error|error:|Traceback" | head -60; exit 1; }
}

for arm in $ARMS; do
  case $arm in
    A) train A --tokenizer qwen $TRAIN ;;
    B) train B --tokenizer extended $TRAIN ;;
    C) TOK=$(python -c "import json;print(json.load(open('runs/A/train.json'))['tokens_trained'])")
       train C --tokenizer extended --tokens $TOK $( [ "$RUN" = smoke ] && echo "--max-steps 20" ) ;;
  esac
done

echo "== evaluate the trained models"
i=0
for arm in $ARMS; do
  ev $arm $((i % NGPU)) --model runs/$arm
  i=$((i + 1))
  if [ $((i % NGPU)) = 0 ]; then wait; fi
done
wait
for f in runs/eval-*.log; do grep -q '"name"' $f || { echo "== $f failed:"; tail -20 $f; }; done

echo "== summary"
python scripts/summarize_m1.py runs
echo "== done: copy from '== lipi M1' to here and send it back"
