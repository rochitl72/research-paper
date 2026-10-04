"""Causal test of the word plan with activation patching between two natural contexts.

Pair: context A where the model writes word X, context B where it writes Y,
with X and Y sharing their first token t1 but differing at their second token.

We run B + [t1] while overwriting the residual stream at B's *pre-word*
position (the last context token) at layer L with A's activation from the
same layer and position. If the model then continues with X's later tokens,
the information that selects X over Y was causally stored at the pre-word
position by layer L, before the first token of the word was written.

Both contexts are real text; no input is ever split or altered.
"""
from __future__ import annotations

import random
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch

from .corpus import SeqRecord
from .models import Loaded, decoder_layers


def mine_pairs(records: list[SeqRecord], min_conf: float = 0.5, ctx_len: int = 64, max_pairs: int = 100, seed: int = 0):
    """Find (A, B) contexts whose continuations the model is confident about."""
    cands = defaultdict(list)
    for rec in records:
        for u in rec.units:
            if u.kind != "word" or u.n_tokens < 2 or u.start < 8:
                continue
            cont = rec.logp[u.start + 1 : u.end]
            if not rec.top1[u.start + 1 : u.end].all() or np.exp(cont.sum()) < min_conf:
                continue
            ctx = rec.ids[max(0, u.start - ctx_len) : u.start]
            cands[u.ids[0]].append(dict(ctx=ctx, word=u.ids))
    rng = random.Random(seed)
    pairs = []
    groups = [g for g, v in cands.items() if len({c["word"] for c in v}) >= 2]
    rng.shuffle(groups)
    # round-robin over groups so no single first token dominates
    while len(pairs) < max_pairs and groups:
        nxt = []
        for g in groups:
            v = cands[g]
            words = defaultdict(list)
            for c in v:
                words[c["word"]].append(c)
            ws = [w for w in words if len(w) >= 2]
            options = [(a, b) for a in ws for b in ws if a != b and a[1] != b[1]]
            if not options:
                continue
            a, b = rng.choice(options)
            A, B = rng.choice(words[a]), rng.choice(words[b])
            key = (tuple(A["ctx"][-8:]), tuple(B["ctx"][-8:]))
            if any(key == p["key"] for p in pairs):
                continue
            pairs.append(dict(A=A, B=B, key=key))
            nxt.append(g)
            if len(pairs) >= max_pairs:
                break
        if not nxt:
            break
        groups = nxt
    return pairs


@contextmanager
def _hooks(lm: Loaded, fn_by_layer: dict):
    """fn_by_layer[L](tensor)->tensor; L=0 is the embedding output, L>=1 is block L-1 output."""
    handles = []
    layers = decoder_layers(lm.model)
    emb = lm.model.get_input_embeddings()

    def wrap(fn):
        def hook(_mod, _inp, out):
            if isinstance(out, tuple):
                return (fn(out[0]),) + tuple(out[1:])
            return fn(out)
        return hook

    for L, fn in fn_by_layer.items():
        mod = emb if L == 0 else layers[L - 1]
        handles.append(mod.register_forward_hook(wrap(fn)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def capture(lm: Loaded, ids: list[int]) -> dict[int, torch.Tensor]:
    """Residual stream at every hook point for one sequence ([T, d] per layer)."""
    store = {}

    def make(L):
        def f(t):
            store[L] = t[0].detach().clone()
            return t
        return f

    with _hooks(lm, {L: make(L) for L in range(lm.n_layers + 1)}):
        lm.model(input_ids=torch.tensor([ids], device=lm.device))
    return store


@torch.no_grad()
def _logits_last(lm: Loaded, ids, patch: dict | None = None):
    with _hooks(lm, patch or {}):
        out = lm.model(input_ids=torch.tensor([ids], device=lm.device))
    return out.logits[0, -1].float()


@torch.no_grad()
def _greedy_with_patch(lm: Loaded, ids, n_new: int, patch: dict | None = None):
    with _hooks(lm, patch or {}):
        out = lm.model(input_ids=torch.tensor([ids], device=lm.device), use_cache=True)
    cache = out.past_key_values
    nxt = int(out.logits[0, -1].argmax())
    gen = [nxt]
    for _ in range(n_new - 1):
        out = lm.model(input_ids=torch.tensor([[nxt]], device=lm.device), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        nxt = int(out.logits[0, -1].argmax())
        gen.append(nxt)
    return gen


def run_pair(lm: Loaded, pair: dict, layers: list[int], where: str = "preword") -> dict:
    A, B = pair["A"], pair["B"]
    t1 = A["word"][0]
    a_ids = A["ctx"] + [t1]
    b_ids = B["ctx"] + [t1]
    x2, y2 = A["word"][1], B["word"][1]
    x_rest, y_rest = list(A["word"][1:]), list(B["word"][1:])
    capA = capture(lm, a_ids)
    pa = len(A["ctx"]) - 1 if where == "preword" else len(a_ids) - 1
    pb = len(B["ctx"]) - 1 if where == "preword" else len(b_ids) - 1

    def diff(logits):
        return float(logits[x2] - logits[y2])

    mA = diff(_logits_last(lm, a_ids))
    mB = diff(_logits_last(lm, b_ids))
    res = dict(mA=mA, mB=mB, layers=[], effect=[], transfer_x=[], keep_y=[])
    for L in layers:
        src = capA[L][pa]

        def f(t, src=src):
            if t.shape[1] > pb:
                t = t.clone()
                t[0, pb] = src.to(t.dtype)
            return t

        patch = {L: f}
        mP = diff(_logits_last(lm, b_ids, patch))
        denom = mA - mB
        res["layers"].append(L)
        res["effect"].append((mP - mB) / denom if abs(denom) > 1e-6 else float("nan"))
        n_new = max(len(x_rest), len(y_rest))
        gen = _greedy_with_patch(lm, b_ids, n_new, patch)
        res["transfer_x"].append(gen[: len(x_rest)] == x_rest)
        res["keep_y"].append(gen[: len(y_rest)] == y_rest)
    return res


def sweep(lm: Loaded, pairs: list[dict], layers: list[int], where: str = "preword", progress=True) -> dict:
    out = []
    for i, p in enumerate(pairs):
        out.append(run_pair(lm, p, layers, where))
        if progress:
            print(f"\r  patching {where} {i+1}/{len(pairs)}", end="", flush=True)
    if progress:
        print()
    eff = np.array([r["effect"] for r in out])
    tx = np.array([r["transfer_x"] for r in out], dtype=float)
    ky = np.array([r["keep_y"] for r in out], dtype=float)
    return dict(
        where=where,
        layers=layers,
        effect_mean=np.nanmean(eff, 0).tolist(),
        effect_median=np.nanmedian(eff, 0).tolist(),
        effect_q25=np.nanpercentile(eff, 25, 0).tolist(),
        effect_q75=np.nanpercentile(eff, 75, 0).tolist(),
        transfer_rate=tx.mean(0).tolist(),
        keep_rate=ky.mean(0).tolist(),
        n_pairs=len(out),
        per_pair=out,
    )


# ---------------------------------------------------------------------------
# Attention knockout: is the pre-word position *necessary* for spelling?
# ---------------------------------------------------------------------------

@torch.no_grad()
def _cont_logp(lm: Loaded, ids: list[int], start: int, end: int, blocked: int | None):
    """Sum log p of tokens start+1..end-1 (the word's continuation) and whether all are greedy-correct.

    With ``blocked`` set, the positions inside the word (start..end-1) cannot attend to that
    context position at any layer; everything else is ordinary causal attention.
    """
    T = len(ids)
    allowed = torch.tril(torch.ones(T, T, dtype=torch.bool, device=lm.device))
    if blocked is not None:
        allowed[start:end, blocked] = False
    dtype = next(lm.model.parameters()).dtype
    # HF takes a 4D mask in additive form: 0 where attention is allowed, a large negative number elsewhere
    mask = torch.zeros(T, T, dtype=dtype, device=lm.device).masked_fill(~allowed, torch.finfo(dtype).min)
    out = lm.model(input_ids=torch.tensor([ids], device=lm.device), attention_mask=mask[None, None])
    lp = torch.log_softmax(out.logits[0, start : end - 1].double() if dtype == torch.float64 else out.logits[0, start : end - 1].float(), -1)
    tgt = torch.tensor(ids[start + 1 : end], device=lm.device)
    return float(lp.gather(-1, tgt[:, None]).sum()), bool((lp.argmax(-1) == tgt).all())


def knockout(lm: Loaded, records: list[SeqRecord], n_words: int = 200, ctx_len: int = 64, seed: int = 0, progress=True) -> dict:
    """Block attention from a word's own positions to one context position and measure the damage.

    Compared positions: the pre-word position (last token before the word), the
    token before it, and a random earlier context position. If the word plan
    were handed over through the pre-word position, blocking it should hurt far
    more than blocking the others.
    """
    rng = random.Random(seed)
    cands = []
    for rec in records:
        for u in rec.units:
            if u.kind == "word" and u.n_tokens >= 3 and u.start >= 12 and rec.top1[u.start + 1 : u.end].all():
                cands.append((rec, u))
    rng.shuffle(cands)
    rows = []
    for i, (rec, u) in enumerate(cands[:n_words]):
        lo = max(0, u.start - ctx_len)
        ids = rec.ids[lo : u.end]
        s, e = u.start - lo, u.end - lo
        base, base_ok = _cont_logp(lm, ids, s, e, None)
        r = dict(n_tokens=u.n_tokens, base=base, base_ok=base_ok)
        for name, pos in (("preword", s - 1), ("preword_minus_1", s - 2), ("random_context", rng.randrange(1, s - 2))):
            lp, ok = _cont_logp(lm, ids, s, e, pos)
            r[name] = lp - base
            r[name + "_ok"] = ok
        rows.append(r)
        if progress:
            print(f"\r  knockout {i + 1}/{min(n_words, len(cands))}", end="", flush=True)
    if progress:
        print()
    out = dict(n_words=len(rows), base_ok=float(np.mean([r["base_ok"] for r in rows])))
    for name in ("preword", "preword_minus_1", "random_context"):
        d = np.array([r[name] for r in rows]) / np.log(2)
        out[name] = dict(mean_delta_bits=float(d.mean()), median_delta_bits=float(np.median(d)),
                         still_spelled=float(np.mean([r[name + "_ok"] for r in rows])))
    out["per_word"] = rows
    return out
