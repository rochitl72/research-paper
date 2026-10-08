# Research log

Chronological record of decisions, surprises and deviations from the proposal
("Words Before Tokens"). Kept so the paper's method and limitations sections can be
written from it.

## 2026-10-04: build and first runs

**Setup.** Code developed and unit-tested in a Linux container (CPU). Experiments run on a
MacBook M1 Pro (16 GB) with PyTorch MPS, float32. Data: 6,000 consecutive Wikipedia
paragraphs each for Tamil, Hindi and English (`wikimedia/wikipedia`, 20231101 dumps);
paragraphs 0-3999 form the fit pool, 4000-5999 the held-out pool. Each run uses the first
1,500 fit and 500 held-out paragraphs, truncated to 256 tokens.

### Surprise 1: the first token of a Tamil or Hindi word carries almost no information

The proposal assumed a word's first token is a real decision and the rest is spelling.
With Qwen3's byte-level BPE this is false for Indic scripts. Nearly every Tamil word
starts with the same token: a space plus the UTF-8 lead bytes shared by the whole Tamil
block. On Qwen3-0.6B only about 8% of a Tamil word's information (in bits) sits in its
first token, against about 95% for English. Tamil words take 9.6 tokens on average, and
36% of the within-word token boundaries fall inside a single UTF-8 character.

Consequences for the design:

- "Words sharing a first token" is almost the whole Tamil vocabulary, so the plan probe
  became "pick the upcoming word among the 1,000 most frequent multi-token words".
- Spelling share at p ≥ 0.9 came out at 28% of word tokens for Tamil, below the 40% the
  proposal set as the bar for kill test 1. The share of continuation tokens the model
  predicts correctly with greedy decoding is 63%, which is what bounds the decoder.
- The interesting quantity is where in the word the information sits; `bits_by_position`
  was added for that.

### Surprise 2: the word is readable before it is written, but that state does not decide it

Linear probes on the pre-word state beat both controls (Tamil 14.5% vs 3.2% most-frequent
and 1.5% shuffled-label; Hindi 39.8% vs 24.4% and 15.1%), and they are far more accurate on
words the model goes on to spell correctly. But activation patching of the pre-word
position moves the model at most 8% (Tamil) to 12% (Hindi) of the way toward the donor
word, while patching the first-token position reaches 100% at the last layer. So the
proposal's hypothesis of a localized pre-word plan that is then executed is **not
supported**: later positions recompute the word from the whole context.

An attention-knockout test (`03b_knockout.py`) was added as a necessity check. It shows the
other half: cutting attention from the word's own positions to the pre-word position costs
2.3 bits (Tamil) and 1.8 bits (Hindi) on the word's continuation and drops correct spelling
from 63% to 28% (Tamil) and 85% to 53% (Hindi), while cutting the token before it or a
random context token does almost nothing. So the pre-word position is a necessary relay,
yet swapping its content does not redirect the word. Working reading: it passes on
something the spelling needs (boundary or local-context information) while word identity
is assembled from many context positions. This needs a finer experiment (path patching per
head) before it goes in a paper as a claim.

### Design changes made along the way

- **In-word states.** Because the word is recomputed at every position, a drafter should
  read the freshest state. `run_corpus` now also caches the final-layer state at the first
  six positions inside each multi-token word, and a "current-state" word head is trained on
  them. A test asserts the cached states equal what the decoder hands to drafters.
- **Decoder emulation.** The first version credited a drafter with one draft per word.
  It was replaced with a replay of the real decoder loop (re-draft after every rejection),
  so emulated numbers and real decoding measure the same thing.
- **Token-ahead heads.** Lexicon coverage on held-out Tamil is only 39% (head lexicon) to
  50% (dictionary), so linear Medusa-style heads that predict the next four tokens from the
  current state were added. They need no lexicon. Combined with the word head they give
  the best result.
- **Tokenizer-only dictionary.** The dictionary baseline is built from all 4,000 fit
  paragraphs with the tokenizer alone, so it is not handicapped by the smaller set of
  paragraphs that went through the model.
- **Probes and heads on the GPU.** The first probe run used CPU and would have taken
  hours for Hindi; training moved to MPS and probe classes were capped at the 1,000 most
  frequent words.
- **Hidden-state storage.** States are kept only for multi-token words, as float16, at
  layers 0, 7, 14, 21, 28 (pre-word) and layer 28 (in-word).
- **4D attention masks.** Hugging Face expects a user-supplied 4D mask in additive form
  (0 = attend, large negative = blocked). Passing a 0/1 mask silently adds 1 to allowed
  logits; a test now checks the explicit causal mask reproduces ordinary attention exactly.

### Things that did not work or are still open

- Experiments and commits are run and pushed from the author's Mac (M1 Pro, 16 GB).
- Wall-clock speedups are measured with batch size 1 on MPS. On CPU the same decoder is
  slower than plain decoding because extra draft tokens are not free there.
- Greedy decoding only. Lossless speculative *sampling* needs the standard
  accept/reject rule and is not implemented.
- One model family (Qwen3). Gemma was planned as the second family and is not run yet.
