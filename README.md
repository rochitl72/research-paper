# TokenLift

**TokenLift: Exploiting Latent Word Plans for Exact Word-Level Speculative Decoding in Byte-BPE Indic Language Models**

How much of what a language model writes in Tamil or Hindi is just spelling, when does the
model know which word it is writing, and can that knowledge make generation faster without
changing the output? This repository holds the paper sources (IEEE Access and arXiv versions),
the `tokenlift` tool, the `wordhead` research code, experiment scripts, raw results, figures and
run logs.

**Status (2026-10-08):** code, tests and both papers complete; mechanistic analysis on Qwen3-0.6B/1.7B
(+ XGLM-564M) and decoder results on Qwen3-0.6B/1.7B/4B for Tamil and Hindi, all run on a MacBook M1 Pro.
Start with [`tokenlift/README.md`](tokenlift/README.md) for the tool or [`docs/results.md`](docs/results.md)
for results. Licensed MIT.

## Headline results (Qwen3-0.6B-Base)

| | Tamil | Hindi | English |
| --- | ---: | ---: | ---: |
| Tokens per word | 9.59 | 4.82 | 1.14 |
| Share of a word's information in its first token | 7.7% | 24.9% | 94.7% |
| Continuation tokens the model gets right greedily | 63.1% | 65.6% | 70.6% |
| Probe picks the upcoming word before it is written (1,000 candidates) | 14.5% | 39.8% | n/a |
| ...most-frequent-word baseline | 3.2% | 24.4% | n/a |
| Patching the pre-word state: shift toward the donor word (best layer, median) | 0.08 | 0.12 | n/a |
| Lossless decoding: tokens per forward pass (plain = 1.00) | 1.88 | 1.66 | n/a |
| Lossless decoding: wall-clock speedup on M1 Pro | ×1.46 | ×1.55 | n/a |

In words: the upcoming word is partly readable from the model's state before any of it is
written, but that state does not decide the word; the word is re-derived from context as it
is spelled. A lossless word-drafting decoder is about 1.5 times faster than plain greedy
decoding, a modest gain over a context-blind dictionary. Two of the proposal's three
kill-test bars were missed; see the verdict table in `docs/results.md`.

## Repository map

| Path | What is there |
| --- | --- |
| `wordhead/` | The library (segmentation, corpus pass, spelling share, probes, patching, word heads, decoder, CLI) |
| `experiments/` | Numbered scripts `00`-`06` and `run_all.sh`; each writes JSON to `results/` and appends to `logs/` |
| `tests/` | 19 unit tests, plus one integration test on the real model (`pytest -m slow`) |
| `results/<model>/` | Raw JSON for every experiment, including per-pair and per-prompt records |
| `figures/<model>/` | PNG figures (200 dpi) generated from `results/` |
| `logs/` | Console output of every run, with environment and arguments |
| `docs/architecture.md` | Data flow, modules, definitions, why the decoder is lossless |
| `docs/results.md` | Findings, verdicts against the pre-set kill tests, limits |
| `docs/results_tables.md` | Every table, generated from `results/` |
| `docs/research_log.md` | Decisions, surprises and deviations from the proposal |
| `data/raw/` | 6,000 Wikipedia paragraphs each for Tamil, Hindi, English |

## Install

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pytest                 # 19 tests, about 20 s; downloads only a tokenizer
pytest -m slow         # real-model lossless check (downloads Qwen3-0.6B, 1.2 GB)
```

Works on Apple Silicon (MPS), CUDA and CPU. On a Mac set `PYTORCH_ENABLE_MPS_FALLBACK=1`.
Use `--dtype float32` for experiments: half precision can flip near-tied argmaxes between
batch shapes, which breaks the exact-match check.

## Reproduce the experiments

```bash
N_FIT=1500 N_EVAL=500 PAIRS=80 experiments/run_all.sh Qwen/Qwen3-0.6B-Base float32
python experiments/06_report.py --models Qwen/Qwen3-0.6B-Base
```

On an M1 Pro the 0.6B run takes about 75 minutes: 14 for the corpus pass, 5 for probes,
13 for heads and decoding, 40 for patching, 3 for knockout. It needs about 2.5 GB of disk
for cached hidden states (`cache/`, not committed).

## Use the tool

```bash
# 1. How much of this model's output in this language is spelling?
wordhead audit --model Qwen/Qwen3-0.6B-Base --text-file data/raw/wiki_ta.jsonl --limit 300

# 2. Build a lexicon and fit a word head from an unlabeled corpus
wordhead fit --model Qwen/Qwen3-0.6B-Base --text-file data/raw/wiki_ta.jsonl --out heads/ta

# 3. Generate with word drafts; prints whether the output equals plain greedy decoding
wordhead generate --model Qwen/Qwen3-0.6B-Base --head-dir heads/ta --prompt "தமிழ்நாடு இந்தியாவின்"
```

The CLI covers the dictionary and pre-word word-head drafters. The best drafter in the
experiments (current-state word head followed by token-ahead heads) is available from
Python (`wordhead.heads.AheadDrafter`) and in `experiments/04_heads_and_decoding.py`; it is
not wired into the CLI yet.

## What is not done

- Second model family (Gemma) and the 4B size.
- Sampling (only greedy decoding is lossless here), MLX and llama.cpp back ends.
- The prior-work check the proposal lists as step 0.
- The paper.
