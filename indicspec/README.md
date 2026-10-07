# IndicSpec

**Lossless word-level speculative decoding for Indic-script LLMs.**

Byte-level BPE tokenizers shatter Indic words into many sub-word tokens — Tamil
averages ~9 tokens per word, Hindi ~5, against ~1.1 for English. Most of those
tokens are *spelling*: once the model has committed to a word, emitting its
remaining pieces is near-deterministic. IndicSpec exploits this with a small
**word-scoped EAGLE head** that drafts the rest of a word in feature space, and
verifies the draft against the target model in a single forward pass. Output is
**identical to the target's own decoding** (greedy), or drawn from the **target's
exact sampling distribution** (temperature > 0) — never an approximation.

## Install

```bash
pip install -e .          # from the repository root (installs wordhead + indicspec)
```

## Quick start

```python
from indicspec import IndicSpec

spec = IndicSpec.from_pretrained("Qwen/Qwen3-0.6B-Base")

# train a word-scoped head per language on a cached corpus (one-time)
spec.fit(langs=["ta", "hi"], epochs=30)
spec.save_head("ta", "head_ta.pt")

# lossless greedy (token-for-token identical to the model's own greedy output)
print(spec.generate("தமிழ்நாட்டின் தலைநகரம்", max_new=64, lang="ta")["text"])

# lossless speculative sampling (preserves the target's T-sampled distribution)
print(spec.sample("தமிழ்நாட்டின் தலைநகரம்", max_new=64, temperature=0.8, lang="ta")["text"])
```

Load a saved head later with `spec.load_head("ta", "head_ta.pt")`.

## Command line

```bash
# how much headroom does a model have on your text? (no head needed)
indicspec report --model Qwen/Qwen3-0.6B-Base --text-file samples_ta.txt

# generate from a trained head (temperature 0 = greedy, >0 = sampling)
indicspec gen --model Qwen/Qwen3-0.6B-Base --lang ta --head head_ta.pt \
              --prompt "தமிழ்நாட்டின் தலைநகரம்" --temperature 0.8
```

## What you get (Qwen3-0.6B-Base, Apple M1 Pro, fp16)

| Language | Decoder                         | tokens / forward | wall-clock |
|----------|---------------------------------|------------------|------------|
| Tamil    | plain                           | 1.00             | ×1.00      |
| Tamil    | dictionary hybrid (baseline)    | 1.89             | ×1.12      |
| Tamil    | **IndicSpec (EAGLE head)**      | **2.15**         | **×1.41**  |
| Hindi    | plain                           | 1.00             | ×1.00      |
| Hindi    | dictionary hybrid (baseline)    | 1.68             | ×1.15      |
| Hindi    | **IndicSpec (EAGLE head)**      | **1.82**         | **×1.40**  |

Speculative **sampling** (temperature 0.7–1.0) keeps essentially the same
tokens-per-forward as greedy; see `results/<model>/sampling.json`.

## Losslessness

Greedy output is **identical to plain greedy decoding by construction**: every
draft token is kept only if it equals the target model's own arg-max at that
position, and rejected positions are cropped from the KV cache. The only
divergences we observe are at floating-point ties — positions where two tokens
share the top logit to within ~1e-2 and the order flips with batch shape; these
are a property of fp arithmetic, not of the method. For temperature > 0 we use
the standard speculative-sampling accept/reject rule (Leviathan et al., 2023;
Chen et al., 2023), which provably preserves the target's sampling distribution.

Comprehension is never touched: the model only ever reads its own original
sub-word tokens. Word scoping affects **generation**, not the context.

## How it works

1. A word boundary is detected in the token stream.
2. The EAGLE head takes the target's hidden state and the current token
   embedding and autoregressively drafts feature vectors for the rest of the
   word, decoding each to a token (greedy) or a full-vocabulary distribution via
   the target's own final norm + LM head (sampling).
3. `[first_token] + draft` is verified in one target forward pass; accepted
   tokens are committed, the KV cache is cropped at the first rejection.

The head is trained with scheduled-sampling rollouts to match its own
multi-step drafting at inference (mitigating exposure bias).

## API

- `IndicSpec.from_pretrained(model_name)` — load a target model.
- `.fit(langs, epochs=30, vocab_size=4000, max_draft=8)` — train a head per language.
- `.save_head(lang, path)` / `.load_head(lang, path)` — persist heads (with vocab).
- `.generate(prompt, max_new, lang, accelerate=True)` — lossless greedy.
- `.sample(prompt, max_new, temperature, lang, seed)` — lossless speculative sampling.
- `indicspec.diagnostic.headroom(lm, texts)` — predict achievable speed-up from
  tokenizer fragmentation and the model's continuation predictability.

Part of the *Words before tokens* study. MIT licensed.
