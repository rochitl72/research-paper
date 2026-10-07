"""Lossless speculative *sampling* for the word-scoped EAGLE head (IndicSpec).

Greedy verification only guarantees the arg-max path. For temperature > 0 we use
the standard speculative-sampling accept/reject rule (Leviathan et al., 2023;
Chen et al., 2023), which provably preserves the target model's sampling
distribution. The draft distribution must be over the *full* vocabulary, so we
feed the EAGLE head's predicted feature through the target model's own final
norm + LM head (EAGLE-style), rather than the head's small word classifier.

Control flow mirrors wordhead.specdec.generate exactly (same KV-cache cropping
and the invariant that ``last_h`` is the hidden state at the position whose
logits produced the current token), so draft features match what the head saw
in training. The only changes are: tokens are sampled instead of arg-maxed, and
acceptance is the stochastic min(1, p/q) rule with residual resampling.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .models import Loaded
from .words import Segmenter


@dataclass
class SampleResult:
    tokens: list[int]
    n_forward: int
    seconds: float
    draft_lengths: list[int] = field(default_factory=list)
    accepted: list[int] = field(default_factory=list)

    @property
    def tokens_per_forward(self) -> float:
        return len(self.tokens) / max(self.n_forward, 1)


def _output_head(lm: Loaded):
    """CPU copies of the target's final RMSNorm + LM head, for full-vocab draft dists."""
    norm = lm.model.model.norm
    w = norm.weight.detach().cpu().float()
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
    Wt = lm.model.lm_head.weight.detach().cpu().float().t().contiguous()  # (d, V)

    def head_logits(f_raw):  # f_raw: (d,) cpu float  -> (V,)
        xn = f_raw / torch.sqrt(f_raw.pow(2).mean(-1, keepdim=True) + eps) * w
        return xn @ Wt
    return head_logits


@torch.no_grad()
def generate_sample(lm: Loaded, prompt_ids, max_new, head, emb, temperature=1.0,
                    segmenter: Segmenter | None = None, stop_ids=None, max_draft=8,
                    layer=-1, seed=0) -> SampleResult:
    _ = segmenter  # boundary info is carried by the head's STOP class; kept for API parity
    stop_ids = stop_ids if stop_ids is not None else ({lm.tokenizer.eos_token_id} - {None})
    T = max(float(temperature), 1e-5)
    g = torch.Generator().manual_seed(seed)
    head = head.to("cpu").eval()
    emb = emb.detach().cpu().float()
    head_logits = _output_head(lm)

    def draft_dist(f_std):  # standardized predicted feature -> full-vocab prob
        f_raw = f_std * head.sd + head.mu
        return torch.softmax(head_logits(f_raw) / T, -1)

    t0 = time.perf_counter()
    out = lm.model(input_ids=torch.tensor([prompt_ids], device=lm.device),
                   use_cache=True, output_hidden_states=True)
    n_forward = 1
    cache = out.past_key_values
    last_logits = out.logits[0, -1].float().cpu()
    last_h = out.hidden_states[layer][0, -1].float().cpu()

    gen: list[int] = []
    pending_tok = None   # forced next token (residual correction); else sample from last_logits
    pending_h = None     # hidden state at the predicting position of pending_tok
    dls, accs = [], []

    while len(gen) < max_new:
        if pending_tok is None:
            t = int(torch.multinomial(torch.softmax(last_logits / T, -1), 1, generator=g))
            h_cur = last_h
        else:
            t, h_cur, pending_tok, pending_h = pending_tok, pending_h, None, None
        gen.append(t)
        if t in stop_ids or len(gen) >= max_new:
            break

        # ---- roll out a within-word draft from (h_cur, t) ----
        draft, qs = [], []
        f_std = head.standardize(h_cur if isinstance(h_cur, torch.Tensor) else torch.as_tensor(h_cur).float())
        last = t
        budget = min(max_draft, max_new - len(gen))
        for _ in range(budget):
            f_next, tok_logits = head(f_std.unsqueeze(0), emb[last].unsqueeze(0))
            if int(tok_logits[0].argmax()) == head.stop:
                break
            q = draft_dist(f_next[0])
            dtok = int(torch.multinomial(q, 1, generator=g))
            draft.append(dtok); qs.append(q)
            f_std = f_next[0]; last = dtok
        if not draft:
            # no speculation this step: refresh last_logits/last_h via a 1-token forward
            out = lm.model(input_ids=torch.tensor([[t]], device=lm.device), past_key_values=cache,
                           use_cache=True, output_hidden_states=True)
            n_forward += 1
            cache = out.past_key_values
            last_logits = out.logits[0, -1].float().cpu()
            last_h = out.hidden_states[layer][0, -1].float().cpu()
            continue

        # ---- verify [t] + draft in one forward ----
        inp = [t] + draft
        out = lm.model(input_ids=torch.tensor([inp], device=lm.device), past_key_values=cache,
                       use_cache=True, output_hidden_states=True)
        n_forward += 1
        cache = out.past_key_values
        plog = out.logits[0].float().cpu()
        hs = out.hidden_states[layer][0].float().cpu()

        k, rej = 0, None
        for i, dtok in enumerate(draft):
            p = torch.softmax(plog[i] / T, -1)
            ratio = min(1.0, float(p[dtok] / qs[i][dtok].clamp_min(1e-9)))
            if float(torch.rand(1, generator=g)) < ratio:
                k += 1
            else:
                resid = torch.clamp(p - qs[i], min=0)
                resid = resid / resid.sum().clamp_min(1e-9)
                rej = int(torch.multinomial(resid, 1, generator=g))
                break

        gen.extend(draft[:k])
        dls.append(len(draft)); accs.append(k)
        if k < len(draft):                       # rejection at position k
            cache.crop(-(len(draft) - k))        # keep t+draft[:k]; drop the rest
            pending_tok = rej                    # correction sampled from the residual
            pending_h = hs[k]                    # state at inp[k] predicts position k+1
        else:                                     # all accepted -> bonus token from target
            last_logits = plog[len(draft)]
            last_h = hs[len(draft)]

    return SampleResult(tokens=gen[:max_new], n_forward=n_forward,
                        seconds=time.perf_counter() - t0, draft_lengths=dls, accepted=accs)
