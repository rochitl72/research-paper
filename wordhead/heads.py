"""Word lexicon, word-level output heads and drafters.

A *word head* is a linear softmax classifier over a lexicon of multi-token
words, reading the hidden state at the position *before* a word starts. With
the base model frozen and features cached, fitting it is a convex problem
(multinomial logistic regression), so it runs on a laptop.

Drafters propose the rest of the current word given the tokens written so far:

* ``DictionaryDrafter`` - most frequent lexicon word with that prefix
                          (context-blind; the n-gram/dictionary baseline)
* ``HeadDrafter``       - highest-scoring lexicon word with that prefix
                          under the word head (reads the model's plan)
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from .corpus import SeqRecord


@dataclass
class Lexicon:
    words: list[tuple[int, ...]]
    freq: list[int]
    texts: list[str]

    def __post_init__(self):
        self.index = {w: i for i, w in enumerate(self.words)}
        self.by_prefix: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for i, w in enumerate(self.words):
            for k in range(1, len(w)):
                self.by_prefix[w[:k]].append(i)

    @classmethod
    def from_records(cls, records: list[SeqRecord], size: int = 5000, min_count: int = 2) -> "Lexicon":
        cnt, text = Counter(), {}
        for rec in records:
            for u in rec.units:
                if u.kind == "word" and u.n_tokens >= 2:
                    cnt[u.ids] += 1
                    text[u.ids] = u.text
        items = [(w, c) for w, c in cnt.most_common() if c >= min_count][:size]
        return cls(words=[w for w, _ in items], freq=[c for _, c in items], texts=[text[w] for w, _ in items])

    def __len__(self):
        return len(self.words)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(dict(words=[list(w) for w in self.words], freq=self.freq, texts=self.texts), f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(words=[tuple(w) for w in d["words"]], freq=d["freq"], texts=d["texts"])


class WordHead(torch.nn.Module):
    """Linear softmax over lexicon words + one OTHER class (index = len(lexicon))."""

    def __init__(self, d_model: int, n_words: int):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.zeros(d_model), requires_grad=False)
        self.sd = torch.nn.Parameter(torch.ones(d_model), requires_grad=False)
        self.lin = torch.nn.Linear(d_model, n_words + 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.lin((h.float() - self.mu) / self.sd)


def head_training_data(records: list[SeqRecord], lexicon: Lexicon, layer: int):
    X, y = [], []
    other = len(lexicon)
    for rec in records:
        H = rec.preword_hidden.get(layer)
        if H is None:
            continue
        for row, ui in enumerate(rec.preword_units):
            u = rec.units[ui]
            X.append(H[row])
            y.append(lexicon.index.get(u.ids, other))
    return np.stack(X).astype(np.float32), np.array(y)


def fit_head(X: np.ndarray, y: np.ndarray, n_words: int, epochs: int = 30, lr: float = 3e-3,
             weight_decay: float = 1e-4, batch_size: int = 1024, seed: int = 0, verbose=False) -> WordHead:
    """Convex fit (multinomial logistic regression) with mini-batch Adam."""
    torch.manual_seed(seed)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y).long()
    head = WordHead(X.shape[1], n_words)
    head.mu.data = Xt.mean(0)
    head.sd.data = Xt.std(0) + 1e-4
    opt = torch.optim.AdamW(head.lin.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            b = perm[i : i + batch_size]
            loss = torch.nn.functional.cross_entropy(head(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        if verbose:
            print(f"  epoch {ep+1}/{epochs} loss {tot / n:.4f}")
    return head.eval()


class DictionaryDrafter:
    name = "dictionary"

    def __init__(self, lexicon: Lexicon):
        self.lex = lexicon

    def draft(self, prefix: tuple[int, ...], h=None) -> list[int]:
        cands = self.lex.by_prefix.get(tuple(prefix))
        if not cands:
            return []
        best = max(cands, key=lambda i: self.lex.freq[i])
        return list(self.lex.words[best][len(prefix):])


class HeadDrafter:
    name = "head"

    def __init__(self, lexicon: Lexicon, head: WordHead):
        self.lex = lexicon
        self.head = head

    @torch.no_grad()
    def scores(self, h) -> np.ndarray:
        ht = h.detach().float().cpu() if torch.is_tensor(h) else torch.from_numpy(np.asarray(h, dtype=np.float32))
        return self.head(ht[None])[0].numpy()

    def draft(self, prefix: tuple[int, ...], h=None) -> list[int]:
        cands = self.lex.by_prefix.get(tuple(prefix))
        if not cands or h is None:
            return []
        s = self.scores(h)
        best = max(cands, key=lambda i: s[i])
        return list(self.lex.words[best][len(prefix):])


def emulated_acceptance(records: list[SeqRecord], drafter, layer: int | None, kinds=("word",)) -> dict:
    """Teacher-forced emulation on natural text.

    For each multi-token word, give the drafter the true first token (as the
    target model would have produced it) and count how many of the drafted
    continuation tokens match the text *and* the model's own greedy choice.
    These are the sequential decoding steps the drafter would have saved.
    """
    saved, cont_total, drafted, words = 0, 0, 0, 0
    exact = 0
    for rec in records:
        H = rec.preword_hidden.get(layer) if layer is not None else None
        row_of = {ui: r for r, ui in enumerate(rec.preword_units)}
        for ui, u in enumerate(rec.units):
            if u.kind not in kinds or u.start < 1 or u.n_tokens < 2:
                continue
            words += 1
            h = H[row_of[ui]] if H is not None and ui in row_of else None
            d = drafter.draft((u.ids[0],), h)
            truth = list(u.ids[1:])
            ok_model = rec.top1[u.start + 1 : u.end]
            k = 0
            for j, tokd in enumerate(d):
                if j < len(truth) and tokd == truth[j] and ok_model[j]:
                    k += 1
                else:
                    break
            saved += k
            cont_total += len(truth)
            drafted += int(bool(d))
            exact += int(k == len(truth))
    return dict(words=words, cont_tokens=cont_total, saved=saved, saved_share_of_cont=saved / max(cont_total, 1),
                draft_rate=drafted / max(words, 1), exact_word_rate=exact / max(words, 1))
