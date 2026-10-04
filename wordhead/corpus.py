"""One teacher-forced pass over a corpus: per-token statistics + optional hidden states.

Everything downstream (spelling share, plan probes, word heads) reads the
records produced here, so the expensive forward passes happen once.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .models import Loaded
from .words import Segmenter, Unit


@dataclass
class SeqRecord:
    ids: list[int]
    units: list[Unit]
    logp: np.ndarray  # [T] log p(token_j | <j); logp[0] = nan
    entropy: np.ndarray  # [T] entropy (nats) of the predictive distribution for token j
    top1: np.ndarray  # [T] bool, greedy argmax == token_j
    # hidden states at the position *before* each word unit, keyed by layer index
    preword_hidden: dict[int, np.ndarray] = field(default_factory=dict)
    preword_units: list[int] = field(default_factory=list)  # index into units
    # hidden states at the first few positions *inside* each of those words (flat, ragged):
    # inword_hidden[L][inword_offset[r] + j] is the state at the word's j-th token, which emits token j+1
    inword_hidden: dict[int, np.ndarray] = field(default_factory=dict)
    inword_offset: list[int] = field(default_factory=list)
    inword_count: list[int] = field(default_factory=list)

    def state_for_prefix(self, layer: int, row: int, j: int):
        """State available when ``j`` tokens of word ``row`` are written (the state that emitted token j-1)."""
        if j == 1:
            H = self.preword_hidden.get(layer)
            return None if H is None else H[row]
        H = self.inword_hidden.get(layer)
        if H is None or j - 2 >= self.inword_count[row]:
            return None
        return H[self.inword_offset[row] + j - 2]


def iter_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


@torch.no_grad()
def run_corpus(
    lm: Loaded,
    texts: list[str],
    max_len: int = 256,
    batch_size: int = 8,
    hidden_layers: list[int] | None = None,
    word_kinds: tuple[str, ...] = ("word",),
    min_word_tokens: int = 2,
    inword_layers: list[int] | None = None,
    inword_positions: int = 6,
    progress: bool = True,
) -> list[SeqRecord]:
    """Teacher-force ``texts`` through the model.

    ``hidden_layers`` uses HF indexing of ``output_hidden_states``: 0 is the
    embedding output, L is the output of the last block. Hidden states are
    kept only at the position before each word of at least
    ``min_word_tokens`` tokens (single-token words have nothing to spell).
    """
    tok = lm.tokenizer
    seg = Segmenter(tok)
    enc = [tok(t, add_special_tokens=False)["input_ids"][:max_len] for t in texts]
    order = np.argsort([-len(e) for e in enc])
    records: list[SeqRecord | None] = [None] * len(enc)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    done = 0
    for batch_idx in iter_batches(list(order), batch_size):
        seqs = [enc[i] for i in batch_idx]
        T = max(len(s) for s in seqs)
        x = torch.full((len(seqs), T), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), T), dtype=torch.long)
        for r, s in enumerate(seqs):
            x[r, : len(s)] = torch.tensor(s)
            mask[r, : len(s)] = 1
        out = lm.model(
            input_ids=x.to(lm.device),
            attention_mask=mask.to(lm.device),
            output_hidden_states=hidden_layers is not None or inword_layers is not None,
        )
        for r, i in enumerate(batch_idx):
            s = seqs[r]
            n = len(s)
            lp = np.full(n, np.nan, dtype=np.float32)
            en = np.full(n, np.nan, dtype=np.float32)
            t1 = np.zeros(n, dtype=bool)
            if n > 1:
                # one row at a time keeps vocab-sized tensors small (151k-entry vocabularies)
                ids_t = torch.tensor(s, device=out.logits.device)
                logprobs = torch.log_softmax(out.logits[r, : n - 1].float(), dim=-1)
                lp[1:] = logprobs.gather(-1, ids_t[1:, None]).squeeze(-1).cpu().numpy()
                en[1:] = (-(logprobs.exp() * logprobs).sum(-1)).cpu().numpy()
                t1[1:] = (logprobs.argmax(-1) == ids_t[1:]).cpu().numpy()
                del logprobs
            units = seg.segment(s)
            rec = SeqRecord(ids=s, units=units, logp=lp, entropy=en, top1=t1)
            pre = [k for k, u in enumerate(units)
                   if u.kind in word_kinds and u.start >= 1 and u.n_tokens >= min_word_tokens]
            if inword_layers is not None:
                rec.preword_units = pre
                idx, off = [], 0
                for k in pre:
                    u = units[k]
                    c = min(inword_positions, u.n_tokens - 2)  # states that still have a token to emit
                    c = max(c, 0)
                    rec.inword_offset.append(off)
                    rec.inword_count.append(c)
                    idx.extend(range(u.start, u.start + c))
                    off += c
                ipos = torch.tensor(idx, dtype=torch.long, device=lm.device)
                for L in inword_layers:
                    h = out.hidden_states[L][r]
                    rec.inword_hidden[L] = h.index_select(0, ipos).to(torch.float16).cpu().numpy() if idx else np.zeros((0, h.shape[-1]), np.float16)
            if hidden_layers is not None:
                pos = torch.tensor([units[k].start - 1 for k in pre], dtype=torch.long, device=lm.device)
                rec.preword_units = pre
                for L in hidden_layers:
                    h = out.hidden_states[L][r]
                    rec.preword_hidden[L] = h.index_select(0, pos).to(torch.float16).cpu().numpy() if len(pre) else np.zeros((0, h.shape[-1]), np.float16)
            records[i] = rec
        del out
        done += len(batch_idx)
        if progress:
            print(f"\r  forward {done}/{len(enc)}", end="", flush=True)
    if progress:
        print()
    return records  # type: ignore[return-value]
