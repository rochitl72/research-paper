"""Word-scoped autoregressive draft head (EAGLE-style), the IndicSpec core.

Unlike the flat linear word/ahead heads, this head rolls out a word's
continuation tokens AUTOREGRESSIVELY at the feature level: from the current
hidden state and the last token's embedding it predicts (i) the next token and
(ii) the next hidden feature, then feeds that predicted feature back in to draft
the following token -- without running the target model. One target forward pass
then verifies the whole drafted span. This directly exploits the paper's finding
that a fragmented word is re-derived as it is written, so conditioning each draft
token on the evolving state matters.

Trained by teacher forcing on the pre-word and in-word hidden states cached by
the corpus pass; fit by gradient descent on frozen features (laptop-friendly).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from .corpus import SeqRecord
from .models import pick_device


# --------------------------- vocabulary ---------------------------
def ahead_vocab(records: list[SeqRecord], size: int = 4000, kinds=("word",)) -> list[int]:
    """Most frequent continuation tokens (the ids a drafter may emit)."""
    c = Counter()
    for r in records:
        for u in r.units:
            if u.kind in kinds and u.n_tokens > 1:
                c.update(u.ids[1:])
    return [t for t, _ in c.most_common(size)]


# --------------------------- model ---------------------------
class WordEagleHead(nn.Module):
    """(standardized feature f, token embedding e) -> (next feature f', token logits).

    Output classes: the continuation vocab, plus OTHER (id not in vocab) and STOP
    (word boundary). Features are standardized; the predicted feature is produced
    and consumed in the standardized space so the autoregressive rollout stays
    self-consistent.
    """

    def __init__(self, d_model: int, d_emb: int, n_vocab: int, hidden: int = 1024):
        super().__init__()
        self.n_out = n_vocab + 2  # + OTHER + STOP
        self.other = n_vocab
        self.stop = n_vocab + 1
        self.register_buffer("mu", torch.zeros(d_model))
        self.register_buffer("sd", torch.ones(d_model))
        self.inp = nn.Linear(d_model + d_emb, hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.to_feat = nn.Linear(hidden, d_model)
        self.to_tok = nn.Linear(hidden, self.n_out)

    def standardize(self, f):
        return (f - self.mu) / self.sd

    def forward(self, f_std, e):
        z = torch.relu(self.inp(torch.cat([f_std, e], -1)))
        z = z + self.mlp(z)
        return self.to_feat(z), self.to_tok(z)  # next feature (std space), logits


# --------------------------- training data ---------------------------
def eagle_training_data(records, layer, vocab, max_steps: int = 6, kinds=("word",)):
    """Teacher-forced pairs: (feature h_{i-1}, token t_i) -> (feature h_i, token t_{i+1}).

    h_0 = pre-word state (predicts t1); h_i = in-word state i-1 (predicts t_{i+1}).
    Returns Fin, Tin (ids), Fout, Y (vocab index / OTHER / STOP).
    """
    vset = {t: i for i, t in enumerate(vocab)}
    OTHER, STOP = len(vocab), len(vocab) + 1
    Fin, Tin, Fout, Y = [], [], [], []
    for r in records:
        if layer not in r.preword_hidden or not r.inword_hidden:
            continue
        L = max(r.inword_hidden)
        pre = r.preword_hidden[layer]
        inw = r.inword_hidden[L]
        for row, ui in enumerate(r.preword_units):
            u = r.units[ui]
            if u.kind not in kinds:
                continue
            ids = u.ids
            off, cnt = r.inword_offset[row], r.inword_count[row]
            # states: h_0 = pre[row]; h_i = inw[off + i - 1] for i>=1
            states = [pre[row].astype(np.float32)] + [inw[off + k].astype(np.float32) for k in range(cnt)]
            n_pairs = min(len(states) - 1, len(ids) - 1, max_steps)
            for i in range(1, n_pairs + 1):
                Fin.append(states[i - 1]); Tin.append(ids[i - 1])
                Fout.append(states[i])
                nxt = ids[i]
                Y.append(vset.get(nxt, OTHER))
            # STOP target at the end of the word if it finished within the cached window
            if len(ids) - 1 <= len(states) - 1 and len(ids) - 1 <= max_steps:
                Fin.append(states[len(ids) - 1]); Tin.append(ids[-1])
                Fout.append(states[len(ids) - 1])  # no next state; predict self
                Y.append(STOP)
    return (np.array(Fin, np.float32), np.array(Tin, np.int64),
            np.array(Fout, np.float32), np.array(Y, np.int64))


def fit_word_eagle(Fin, Tin, Fout, Y, emb, n_vocab, hidden=1024, epochs=25, lr=2e-3,
                   batch=4096, feat_weight=0.5, seed=0, device=None):
    """emb: (V, d_emb) token embedding matrix (frozen). Trains CE(token) + MSE(feature)."""
    torch.manual_seed(seed)
    dev = device or pick_device()
    d_model = Fin.shape[1]
    emb = emb.detach().to(dev).float()
    head = WordEagleHead(d_model, emb.shape[1], n_vocab, hidden).to(dev)
    Ft = torch.from_numpy(Fin).to(dev)
    head.mu.copy_(Ft.mean(0)); head.sd.copy_(Ft.std(0).clamp_min(1e-4))
    Fin_s = head.standardize(Ft)
    Fout_s = head.standardize(torch.from_numpy(Fout).to(dev))
    Tin_t = torch.from_numpy(Tin).to(dev)
    Y_t = torch.from_numpy(Y).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    n = len(Y)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            e = emb[Tin_t[idx]]
            f_pred, logits = head(Fin_s[idx], e)
            loss = ce(logits, Y_t[idx]) + feat_weight * nn.functional.mse_loss(f_pred, Fout_s[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  eagle epoch {ep}: loss {tot / n:.4f}")
    head.eval()
    return head


# --------------------------- drafter ---------------------------
class WordEagleDrafter:
    """Autoregressive word-scoped drafter that plugs into specdec.generate."""
    name = "head"  # tells specdec.generate to supply the per-step hidden state

    def __init__(self, head: WordEagleHead, emb, vocab, max_draft: int = 10, primary=None, device="cpu"):
        # the per-step autoregressive rollout runs on CPU: tiny matmuls, no GPU
        # dispatch overhead, and it avoids MPS hangs on many small ops.
        self.dev = torch.device(device)
        self.head = head.to(self.dev).eval()
        self.emb = emb.detach().to(self.dev).float()
        self.vocab = vocab
        self.id_of = {i: t for i, t in enumerate(vocab)}
        self.max_draft = max_draft
        self.primary = primary  # optional fallback drafter

    @torch.no_grad()
    def draft(self, prefix, h=None, h_cur=None):
        f = h_cur if h_cur is not None else h
        if f is None or not len(prefix):
            return []
        if isinstance(f, torch.Tensor):
            f = f.detach().to(self.dev, torch.float32).reshape(-1)
        else:
            f = torch.as_tensor(np.asarray(f), dtype=torch.float32, device=self.dev).reshape(-1)
        f_std = self.head.standardize(f)
        last = int(prefix[-1])
        out = []
        for _ in range(self.max_draft):
            e = self.emb[last]
            f_next, logits = self.head(f_std.unsqueeze(0), e.unsqueeze(0))
            k = int(logits[0].argmax())
            if k == self.head.stop or k == self.head.other:
                break
            tok = self.id_of[k]
            out.append(tok)
            f_std = f_next[0]
            last = tok
        if not out and self.primary is not None:
            return self.primary.draft(prefix, h, h_cur)
        return out

    def has_candidates(self, prefix):
        return True

    def observe(self, committed):
        if self.primary is not None and hasattr(self.primary, "observe"):
            self.primary.observe(committed)


# --------------------- rollout (scheduled-sampling) training ---------------------
def eagle_seq_data(records, layer, vocab, K: int = 6, kinds=("word",)):
    """Per-word sequences for rollout training.

    ST[N,K+1,d]: states s_0..s_K (s_0 = pre-word; s_k = in-word k-1), zero-padded.
    TOK[N,K]:   input token at each step (t_1..t_K).
    Y[N,K]:     target class for each step (vocab idx of t_{k+2}, OTHER, or STOP).
    TM[N,K]:    1 if the step has a valid token target.
    FM[N,K]:    1 if the step has a valid feature target (s_{k+1} real, not STOP).
    """
    vset = {t: i for i, t in enumerate(vocab)}
    OTHER, STOP = len(vocab), len(vocab) + 1
    d = records[0].preword_hidden[layer].shape[1]
    ST, TOK, Y, TM, FM = [], [], [], [], []
    for r in records:
        if layer not in r.preword_hidden or not r.inword_hidden:
            continue
        L = max(r.inword_hidden)
        pre, inw = r.preword_hidden[layer], r.inword_hidden[L]
        for row, ui in enumerate(r.preword_units):
            u = r.units[ui]
            if u.kind not in kinds or u.n_tokens < 2:
                continue
            ids = u.ids
            off, cnt = r.inword_offset[row], r.inword_count[row]
            states = [pre[row].astype(np.float32)] + [inw[off + k].astype(np.float32) for k in range(cnt)]
            st = np.zeros((K + 1, d), np.float32)
            tok = np.zeros(K, np.int64)
            y = np.zeros(K, np.int64)
            tm = np.zeros(K, np.float32)
            fm = np.zeros(K, np.float32)
            for k in range(K):
                if k >= len(ids):
                    break
                st[k] = states[min(k, len(states) - 1)]
                tok[k] = ids[k]
                if k + 1 < len(ids):  # predict next real token
                    y[k] = vset.get(ids[k + 1], OTHER); tm[k] = 1.0
                    if k + 1 < len(states):
                        st[k + 1] = states[k + 1]; fm[k] = 1.0
                elif k + 1 == len(ids):  # word ends -> STOP
                    y[k] = STOP; tm[k] = 1.0
            ST.append(st); TOK.append(tok); Y.append(y); TM.append(tm); FM.append(fm)
    return (np.array(ST), np.array(TOK), np.array(Y), np.array(TM), np.array(FM))


def fit_word_eagle_rollout(ST, TOK, Y, TM, FM, emb, n_vocab, hidden=1024, epochs=30, lr=2e-3,
                           batch=2048, feat_weight=0.5, ss_max=0.5, seed=0, device=None):
    """Train with scheduled sampling: the head is fed its OWN predicted feature as
    the next input with probability that anneals from 0 to ``ss_max``, so it learns
    to correct the drift that hurts a multi-step rollout at draft time."""
    torch.manual_seed(seed)
    dev = device or pick_device()
    N, Kp1, d = ST.shape
    K = Kp1 - 1
    emb = emb.detach().to(dev).float()
    head = WordEagleHead(d, emb.shape[1], n_vocab, hidden).to(dev)
    STt = torch.from_numpy(ST).to(dev)
    flat = STt.reshape(-1, d)
    nz = flat.abs().sum(-1) > 0
    head.mu.copy_(flat[nz].mean(0)); head.sd.copy_(flat[nz].std(0).clamp_min(1e-4))
    ST_s = head.standardize(STt)
    TOKt = torch.from_numpy(TOK).to(dev)
    Yt = torch.from_numpy(Y).to(dev)
    TMt = torch.from_numpy(TM).to(dev)
    FMt = torch.from_numpy(FM).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(reduction="none")
    for ep in range(epochs):
        ss = ss_max * ep / max(epochs - 1, 1)
        perm = torch.randperm(N, device=dev)
        tot = tok_ok = tok_n = 0.0
        for s in range(0, N, batch):
            idx = perm[s:s + batch]
            f = ST_s[idx, 0]
            loss = 0.0
            for k in range(K):
                e = emb[TOKt[idx, k]]
                f_pred, logits = head(f, e)
                tmask = TMt[idx, k]
                lt = (ce(logits, Yt[idx, k]) * tmask).sum() / tmask.sum().clamp_min(1)
                fmask = FMt[idx, k].unsqueeze(-1)
                lf = ((f_pred - ST_s[idx, k + 1]) ** 2 * fmask).sum() / fmask.sum().clamp_min(1)
                loss = loss + lt + feat_weight * lf
                with torch.no_grad():
                    tok_ok += ((logits.argmax(-1) == Yt[idx, k]) * tmask).sum().item()
                    tok_n += tmask.sum().item()
                # scheduled sampling: next input feature
                teacher = ST_s[idx, k + 1]
                use_pred = (torch.rand(len(idx), 1, device=dev) < ss).float()
                f = use_pred * f_pred.detach() + (1 - use_pred) * teacher
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  eagle-rollout epoch {ep}: loss {tot / N:.3f}  ss={ss:.2f}  teacher-acc {tok_ok / max(tok_n,1):.3f}")
    head.eval()
    return head
