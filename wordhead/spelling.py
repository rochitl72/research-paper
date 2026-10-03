"""Spelling share: how many output tokens are near-deterministic continuations
of a word the model has already started.

All quantities come from the model's own teacher-forced probabilities on
natural text, so nothing here depends on unnatural inputs.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from .corpus import SeqRecord

LN2 = math.log(2)


def word_table(records: list[SeqRecord], kinds=("word",)) -> list[dict]:
    """One row per word occurrence (excluding sequence-initial words)."""
    rows = []
    for si, rec in enumerate(records):
        for ui, u in enumerate(rec.units):
            if u.kind not in kinds or u.start < 1:
                continue
            lp = rec.logp[u.start : u.end]
            rows.append(
                dict(
                    seq=si,
                    unit=ui,
                    text=u.text,
                    ids=u.ids,
                    n_tokens=u.n_tokens,
                    first_logp=float(lp[0]),
                    cont_logp=lp[1:].astype(float).tolist(),
                    cont_top1=rec.top1[u.start + 1 : u.end].tolist(),
                    cont_entropy=rec.entropy[u.start + 1 : u.end].astype(float).tolist(),
                    boundaries=list(u.boundaries),
                    word_logp=float(lp.sum()),
                )
            )
    return rows


def spelling_summary(rows: list[dict], taus=(0.5, 0.9, 0.99)) -> dict:
    n_words = len(rows)
    n_tok = sum(r["n_tokens"] for r in rows)
    n_cont = n_tok - n_words
    out = dict(n_words=n_words, n_word_tokens=n_tok, n_cont_tokens=n_cont)
    out["tokens_per_word"] = n_tok / max(n_words, 1)
    cont_p = np.exp(np.array([p for r in rows for p in r["cont_logp"]], dtype=float))
    cont_top1 = np.array([t for r in rows for t in r["cont_top1"]], dtype=bool)
    btypes = [b for r in rows for b in r["boundaries"]]
    out["cont_share_of_word_tokens"] = n_cont / max(n_tok, 1)
    for tau in taus:
        k = int((cont_p >= tau).sum())
        out[f"spelling_share@{tau}"] = k / max(n_tok, 1)  # share of ALL word tokens
        out[f"cont_deterministic@{tau}"] = k / max(n_cont, 1)  # share of continuation tokens
    out["cont_greedy_correct"] = float(cont_top1.mean()) if len(cont_top1) else float("nan")
    # information split (bits per word)
    first_bits = -np.array([r["first_logp"] for r in rows]) / LN2
    cont_bits = -np.array([sum(r["cont_logp"]) for r in rows]) / LN2
    out["bits_first_per_word"] = float(first_bits.mean())
    out["bits_cont_per_word"] = float(cont_bits.mean())
    out["share_bits_in_first_token"] = float(first_bits.sum() / max(first_bits.sum() + cont_bits.sum(), 1e-9))
    # breakdown by boundary type
    by = defaultdict(list)
    for b, p in zip(btypes, cont_p):
        by[b].append(p)
    out["boundary_counts"] = {b: len(v) for b, v in by.items()}
    out["boundary_share_of_cont"] = {b: len(v) / max(n_cont, 1) for b, v in by.items()}
    out["boundary_deterministic@0.9"] = {b: float(np.mean(np.array(v) >= 0.9)) for b, v in by.items()}
    # multi-token words only
    multi = [r for r in rows if r["n_tokens"] > 1]
    out["share_words_multitoken"] = len(multi) / max(n_words, 1)
    return out


def by_length(rows: list[dict], tau=0.9, max_len=12) -> dict[int, dict]:
    """Continuation determinism as a function of word length (in tokens)."""
    groups = defaultdict(list)
    for r in rows:
        if r["n_tokens"] > 1:
            groups[min(r["n_tokens"], max_len)].append(r)
    res = {}
    for L, rs in sorted(groups.items()):
        p = np.exp(np.array([x for r in rs for x in r["cont_logp"]]))
        res[L] = dict(n=len(rs), cont_det=float((p >= tau).mean()), share_bits_first=float(
            np.sum([-r["first_logp"] for r in rs]) / max(np.sum([-r["word_logp"] for r in rs]), 1e-9)))
    return res


def frequency_matched_by_length(rows: list[dict], tau=0.9, n_bins=5, min_per_cell=20) -> list[dict]:
    """Same as ``by_length`` but within corpus-frequency bins, so that
    'long word' is not confounded with 'rare word'."""
    freq = Counter(r["text"] for r in rows)
    multi = [r for r in rows if r["n_tokens"] > 1]
    if not multi:
        return []
    f = np.array([math.log(freq[r["text"]]) for r in multi])
    edges = np.quantile(f, np.linspace(0, 1, n_bins + 1))
    out = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        sel = [r for r, x in zip(multi, f) if (lo <= x <= hi if b == n_bins - 1 else lo <= x < hi)]
        groups = defaultdict(list)
        for r in sel:
            groups[min(r["n_tokens"], 12)].append(r)
        for L, rs in sorted(groups.items()):
            if len(rs) < min_per_cell:
                continue
            p = np.exp(np.array([x for r in rs for x in r["cont_logp"]]))
            out.append(dict(freq_bin=b, n_tokens=L, n=len(rs), cont_det=float((p >= tau).mean())))
    return out
