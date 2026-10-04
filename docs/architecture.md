# Architecture

## What the system does

`wordhead` answers three questions about a causal language model and a language whose
script the tokenizer fragments heavily (Tamil, Hindi):

1. **Measurement.** How many of the tokens the model writes are spent spelling out words,
   and how predictable is that spelling?
2. **Mechanism.** When does the model "know" which word it is writing: before the word starts,
   or only as the pieces accumulate? Where is that knowledge stored?
3. **Method.** Can a small linear head that reads the model's own hidden state propose whole
   words, so that the model writes them in fewer sequential forward passes without changing
   its output?

## Data flow

```
 raw text (Wikipedia paragraphs, data/raw/wiki_{ta,hi,en}.jsonl)
        │
        ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ corpus.run_corpus   one teacher-forced pass per paragraph     │
 │   words.Segmenter   token ids → word units + boundary types   │
 │   per token:  log p, entropy, greedy-correct flag             │
 │   per multi-token word:                                       │
 │     pre-word hidden state (several layers)                    │
 │     in-word hidden states (final layer, first 6 positions)    │
 └──────────────────────────────────────────────────────────────┘
        │  SeqRecord list, cached to disk (cache/<model>/<lang>_<split>.pkl)
        ├─────────────► spelling.py   spelling share, bits by position, boundary mix
        ├─────────────► probe.py      plan probes by layer; word identity by tokens written
        ├─────────────► patching.py   causal patching between natural contexts (needs the model)
        └─────────────► heads.py      Lexicon → WordHead (convex fit) → drafters → emulation
                                           │
                                           ▼
                                      specdec.generate   lossless word-drafted greedy decoding
```

The expensive part (forward passes) happens once in `run_corpus`. Everything else except
patching and real decoding runs from the cache on CPU.

## Modules

| Module | Responsibility | Key objects |
| --- | --- | --- |
| `wordhead/models.py` | Load a HF causal LM on MPS, CUDA or CPU; locate decoder blocks | `load`, `Loaded`, `decoder_layers` |
| `wordhead/words.py` | Token bytes table; segment ids into word units; label each within-word boundary as `byte`, `grapheme` or `subword` | `Segmenter`, `Unit` |
| `wordhead/corpus.py` | Teacher-forced pass; per-token statistics; pre-word and in-word hidden states | `run_corpus`, `SeqRecord` |
| `wordhead/spelling.py` | Spelling share and information profile of words | `word_table`, `spelling_summary`, `bits_by_position` |
| `wordhead/probe.py` | Linear probes with majority and shuffled-label controls | `probe_all_layers`, `position_probe` |
| `wordhead/patching.py` | Mine context pairs; patch the residual stream; measure shift toward the donor word | `mine_pairs`, `sweep` |
| `wordhead/heads.py` | Lexicon, word head, drafters, decoder emulation | `Lexicon`, `WordHead`, `fit_head`, `DictionaryDrafter`, `HeadDrafter`, `emulated_steps` |
| `wordhead/specdec.py` | Lossless greedy decoding with word drafts and KV-cache rollback | `generate`, `DecodeResult` |
| `wordhead/cli.py` | `wordhead audit / fit / generate` | |

## Definitions used throughout

- **Word unit.** A maximal run of tokens that starts at whitespace or punctuation. Only units made
  purely of letters and combining marks count as words.
- **Continuation token.** Any token of a word after its first.
- **Spelling share @τ.** Share of word tokens that are continuation tokens to which the model
  assigns probability ≥ τ under teacher forcing.
- **Boundary type.** Why the tokenizer split inside a word: `byte` (inside a UTF-8 code point),
  `grapheme` (before a combining mark, i.e. inside one written syllable), `subword` (between whole
  graphemes).
- **Pre-word state.** Residual stream at the last token before a word, at a given layer
  (HF `hidden_states` indexing: 0 = embeddings, L = output of the last block).
- **Current state.** The state that emitted the most recently written token. With `j` tokens of the
  word written, this is the pre-word state for `j = 1` and the state at the word's `(j-2)`-th token
  otherwise. It is exactly what the decoder has in hand when it asks for a draft; a test asserts
  the cached states and the decoder's states are the same tensors.

## The decoder (why it is lossless)

```
loop:
  t      = argmax of the target model's last logits            # the model's own next token
  draft  = drafter(prefix of the current word incl. t, states)  # rest of the word, or nothing
  logits = model([t] + draft)                                   # ONE forward pass, KV cache reused
  keep draft[i] while draft[i] == argmax(logits[i])             # verify against the model itself
  roll the KV cache back over rejected positions
```

Every kept token equals the model's greedy choice given the tokens before it, so the output is
token-for-token the output of plain greedy decoding. The input side is untouched: the model
always reads its original subword tokens. Tests check equality against `model.generate` on a
random-weight model (float64) with random, perfect and corrupted drafters, and on the real
0.6B model.

## Word head

A `WordHead` is `softmax(W · standardize(h) + b)` over the lexicon words plus one OTHER class.
The base model is frozen and features are cached, so fitting is multinomial logistic regression
(convex) with mini-batch AdamW. Two variants share the code:

- **pre-word head**: trained on pre-word states only; one evaluation per word.
- **current-state head**: trained on the pre-word state and the in-word states; re-evaluated each
  time the decoder asks for a draft, always restricted to lexicon words that match the written prefix.

Both fall back to a frequency dictionary over a larger tokenizer-only lexicon when no head word
matches the prefix.

## Experiment scripts

| Script | Output | Needs the model |
| --- | --- | --- |
| `experiments/00_corpus_pass.py` | cache + `results/<model>/corpus_pass_meta.json` | yes |
| `experiments/01_spelling_share.py` | `spelling_share.json` | no |
| `experiments/02_plan_probe.py` | `plan_probe.json` | no |
| `experiments/03_patching.py` | `patching.json` | yes |
| `experiments/04_heads_and_decoding.py` | `heads_and_decoding.json` | yes (for real decoding) |
| `experiments/05_figures.py` | `figures/<model>/*.png` | no |
| `experiments/run_all.sh` | all of the above for one model | yes |

Every script appends its console output to `logs/<script>_<model>.log`.

## Splits and leakage

`data/raw/wiki_<lang>.jsonl` holds 6,000 consecutive paragraphs per language. Paragraphs 0-3999
are the fit pool and 4000-5999 the held-out pool, so the two come from different articles.
Lexicons and heads are built from the fit pool only; decoder emulation and real decoding use the
held-out pool. Probe train/test splits are grouped by paragraph.
