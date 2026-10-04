"""Lossless greedy decoding with word-level drafts.

Each step:
1. The target model's own greedy token ``t`` is taken from the last logits
   (always correct by definition).
2. If ``t`` belongs to a word, a drafter proposes the rest of that word.
3. ``[t] + draft`` is fed through the target in ONE forward pass; draft tokens
   are kept only while they equal the target's greedy choice. Rejected
   positions are cropped from the KV cache.

The output is token-for-token identical to plain greedy decoding (this is
checked in ``tests/test_specdec.py``). Inputs are always the model's original
subword tokens, so comprehension is never altered.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .models import Loaded
from .words import Segmenter


@dataclass
class DecodeResult:
    tokens: list[int]
    n_forward: int
    seconds: float
    draft_lengths: list[int] = field(default_factory=list)
    accepted: list[int] = field(default_factory=list)

    @property
    def tokens_per_forward(self) -> float:
        return len(self.tokens) / max(self.n_forward, 1)


@torch.no_grad()
def generate(lm: Loaded, prompt_ids: list[int], max_new: int, drafter=None, layer: int = -1,
             segmenter: Segmenter | None = None, stop_ids: set[int] | None = None) -> DecodeResult:
    seg = segmenter or Segmenter(lm.tokenizer)
    stop_ids = stop_ids if stop_ids is not None else ({lm.tokenizer.eos_token_id} - {None})
    need_h = drafter is not None and getattr(drafter, "name", "") == "head"
    t0 = time.perf_counter()
    out = lm.model(input_ids=torch.tensor([prompt_ids], device=lm.device), use_cache=True, output_hidden_states=need_h)
    n_forward = 1
    cache = out.past_key_values
    last_logits = out.logits[0, -1]
    last_h = out.hidden_states[layer][0, -1] if need_h else None
    cache_len = len(prompt_ids)
    gen: list[int] = []
    prev = prompt_ids[-1]
    prefix: list[int] = []
    pre_h = None
    draft_lengths, accepted_counts = [], []
    while len(gen) < max_new:
        t = int(last_logits.argmax())
        starts = seg._starts_unit(t) or seg._ends_with_break(prev)
        if starts:
            prefix, pre_h = [t], last_h
        else:
            prefix = prefix + [t]
        if t in stop_ids:
            gen.append(t)
            break
        if drafter is not None and hasattr(drafter, "observe"):
            drafter.observe(gen + [t])  # optional hook: tokens committed so far
        draft = drafter.draft(tuple(prefix), pre_h, last_h) if drafter is not None else []
        draft = draft[: max(0, max_new - len(gen) - 1)]
        inp = [t] + draft
        out = lm.model(input_ids=torch.tensor([inp], device=lm.device), past_key_values=cache, use_cache=True,
                       output_hidden_states=need_h)
        n_forward += 1
        cache = out.past_key_values
        greedy = out.logits[0].argmax(-1)
        k = 0
        for j, d in enumerate(draft):
            if int(greedy[j]) == d:
                k += 1
            else:
                break
        kept = inp[: 1 + k]
        gen.extend(kept)
        prefix = prefix + draft[:k]
        prev = kept[-1]
        cache_len += 1 + k
        if k < len(draft):
            cache.crop(-(len(draft) - k))  # drop rejected draft positions
        last_logits = out.logits[0, k]
        last_h = out.hidden_states[layer][0, k] if need_h else None
        if draft:
            draft_lengths.append(len(draft))
            accepted_counts.append(k)
    return DecodeResult(tokens=gen[:max_new], n_forward=n_forward, seconds=time.perf_counter() - t0,
                        draft_lengths=draft_lengths, accepted=accepted_counts)
