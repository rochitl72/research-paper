#!/usr/bin/env bash
# Full pipeline for one model. Usage: experiments/run_all.sh Qwen/Qwen3-0.6B-Base [float32]
set -euo pipefail
MODEL=${1:-Qwen/Qwen3-0.6B-Base}
DTYPE=${2:-float32}
N_FIT=${N_FIT:-1000}
N_EVAL=${N_EVAL:-400}
PAIRS=${PAIRS:-80}
cd "$(dirname "$0")"
python 00_corpus_pass.py --model "$MODEL" --dtype "$DTYPE" --langs ta hi en --n-fit "$N_FIT" --n-eval "$N_EVAL"
python 01_spelling_share.py --model "$MODEL"
python 02_plan_probe.py --model "$MODEL" --langs ta hi
python 03_patching.py --model "$MODEL" --dtype "$DTYPE" --langs ta hi --pairs "$PAIRS" --min-conf 0.3
python 04_heads_and_decoding.py --model "$MODEL" --dtype "$DTYPE" --langs ta hi
python 05_figures.py --models "$MODEL"
