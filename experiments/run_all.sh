#!/usr/bin/env bash
# Full pipeline for one model on this machine. Usage: experiments/run_all.sh MODEL [DTYPE]
set -euo pipefail
MODEL=${1:-Qwen/Qwen3-0.6B-Base}
DTYPE=${2:-float32}
N_FIT=${N_FIT:-1500}
N_EVAL=${N_EVAL:-500}
PAIRS=${PAIRS:-80}
SKIP_PASS=${SKIP_PASS:-0}
cd "$(dirname "$0")"
if [ "$SKIP_PASS" != "1" ]; then
  python 00_corpus_pass.py --model "$MODEL" --dtype "$DTYPE" --langs ta hi en --n-fit "$N_FIT" --n-eval "$N_EVAL"
fi
python 01_spelling_share.py --model "$MODEL"
python 02_plan_probe.py --model "$MODEL" --langs ta hi
python 04_heads_and_decoding.py --model "$MODEL" --dtype "$DTYPE" --langs ta hi
python 03_patching.py --model "$MODEL" --dtype "$DTYPE" --langs ta hi --pairs "$PAIRS" --min-conf 0.3
python 03b_knockout.py --model "$MODEL" --dtype "$DTYPE" --langs ta hi --words "${KO_WORDS:-300}"
python 05_figures.py --models "$MODEL"
