#!/bin/bash
# M2 on Kaggle (one T4 is enough; Internet on). In a notebook cell:
#   !cd /tmp && rm -rf lipi && git clone -q --depth 1 https://github.com/Shakhtar-Sankur/lipi && bash lipi/scripts/kaggle_m2.sh
# About 30-40 minutes. QUICK=1 for a 3-minute check.
set -eo pipefail
cd "$(dirname "$0")/.."
echo "== lipi M2: lipi $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python - <<'EOF'
import os, urllib.request
from lipi import corpus, flores
d = os.path.join(corpus.CACHE, "models", "qwen25-0.5b"); os.makedirs(d, exist_ok=True)
for f in ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
    if not os.path.exists(os.path.join(d, f)):
        urllib.request.urlretrieve(f"https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/060db6499f32faf8b98477b0a26969ef7d8b9987/{f}", os.path.join(d, f))
flores.root()
if not os.path.exists(corpus.path("ory_Orya", "heldout.jsonl")):
    corpus.fetch("ory_Orya", groups=0, every=20, holdout_groups=1)
EOF
CUDA_VISIBLE_DEVICES=0 python scripts/speed_m2.py $( [ -n "$QUICK" ] && echo --quick ) --out runs/speed.json 2>&1 | grep -E '^\{|Error|Traceback'
[ -d /kaggle/working ] && mkdir -p /kaggle/working/lipi-m2 && cp runs/speed.json /kaggle/working/lipi-m2/
echo "== summary"
python scripts/summarize_m2.py runs/speed.json
echo "== done: copy from '== lipi M2' to here and send it back"
