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
    def from_texts(cls, tokenizer, texts: list[str], size: int = 20000, min_count: int = 2, max_len: int = 256) -> "Lexicon":
        """Tokenizer-only lexicon (no model pass), so it can cover far more text."""
        from .words import Segmenter

        seg = Segmenter(tokenizer)
        cnt, text = Counter(), {}
        for t in texts:
            ids = tokenizer(t, add_special_tokens=False)["input_ids"][:max_len]
            for u in seg.segment(ids)[:-1]:  # the last unit may be truncated
                if u.kind == "word" and u.n_tokens >= 2:
                    cnt[u.ids] += 1
                    text[u.ids] = u.text
        items = [(w, c) for w, c in cnt.most_common() if c >= min_count][:size]
        return cls(words=[w for w, _ in items], freq=[c for _, c in items], texts=[text[w] for w, _ in items])

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


def head_training_data(records: list[SeqRecord], lexicon: Lexicon, layer: int, mode: str = "preword"):
    """Features and labels for a word head.

    ``preword``: one sample per multi-token word, the state before the word.
    ``current``: additionally the states inside the word (after 1, 2, ... tokens
    are written), so one position-agnostic head can re-draft as the word unfolds.
    Multi-token words outside the lexicon are labelled OTHER.
    """
    X, y = [], []
    other = len(lexicon)
    for rec in records:
        H = rec.preword_hidden.get(layer)
        if H is None:
            continue
        for row, ui in enumerate(rec.preword_units):
            u = rec.units[ui]
            label = lexicon.index.get(u.ids, other)
            X.append(H[row])
            y.append(label)
            if mode == "current" and layer in rec.inword_hidden:
                o, c = rec.inword_offset[row], rec.inword_count[row]
                for j in range(c):
                    X.append(rec.inword_hidden[layer][o + j])
                    y.append(label)
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
    """Context-blind baseline: the most frequent lexicon word with this prefix."""
    name = "dictionary"

    def __init__(self, lexicon: Lexicon):
        self.lex = lexicon
        self._best = {p: max(c, key=lambda i: lexicon.freq[i]) for p, c in lexicon.by_prefix.items()}

    def draft(self, prefix: tuple[int, ...], h=None, h_cur=None) -> list[int]:
        best = self._best.get(tuple(prefix))
        if best is None:
            return []
        return list(self.lex.words[best][len(prefix):])


class HeadDrafter:
    """Ranks the lexicon words that match the written prefix with a word head.

    ``mode="preword"`` reads only the state before the word (one evaluation per
    word). ``mode="current"`` reads the freshest state, the one that emitted
    the latest token. Falls back to ``fallback`` (e.g. a dictionary over a
    larger lexicon) when no head word matches the prefix.
    """
    name = "head"

    def __init__(self, lexicon: Lexicon, head: WordHead, mode: str = "preword", fallback=None):
        self.lex, self.head, self.mode, self.fallback = lexicon, head, mode, fallback
        self._h = None
        self._scores = None

    @torch.no_grad()
    def scores(self, h) -> np.ndarray:
        if h is not self._h:
            ht = h.detach().float().cpu() if torch.is_tensor(h) else torch.from_numpy(np.asarray(h, dtype=np.float32))
            self._scores = self.head(ht[None])[0].numpy()
            self._h = h
        return self._scores

    def draft(self, prefix: tuple[int, ...], h=None, h_cur=None) -> list[int]:
        prefix = tuple(prefix)
        cands = self.lex.by_prefix.get(prefix)
        feat = h_cur if (self.mode == "current" and h_cur is not None) else h
        if not cands or feat is None:
            return self.fallback.draft(prefix) if self.fallback is not None else []
        s = self.scores(feat)
        best = cands[int(np.argmax(s[cands]))]
        return list(self.lex.words[best][len(prefix):])


class OracleDrafter:
    """Upper bound for emulation: always drafts the true remainder of the word."""
    name = "oracle"

    def __init__(self):
        self.truth: tuple[int, ...] = ()

    def draft(self, prefix, h=None, h_cur=None):
        return list(self.truth[len(prefix):])


def emulated_steps(records: list[SeqRecord], drafter, layer: int | None, kinds=("word",)) -> dict:
    """Teacher-forced emulation of the lossless decoder on natural text.

    For each multi-token word the decoder loop is replayed: the target model
    emits one token per forward pass; after each pass the drafter proposes the
    rest of the word from the tokens written so far, and drafted tokens are
    accepted while they equal the text *and* the model's own greedy choice
    (so acceptance is what greedy verification would accept). Returns forward
    passes needed per word token (1.0 = no saving).
    """
    steps = tokens = words = exact = drafted_tok = accepted_tok = 0
    for rec in records:
        row_of = {ui: r for r, ui in enumerate(rec.preword_units)}
        for ui, u in enumerate(rec.units):
            if u.kind not in kinds or u.start < 1 or u.n_tokens < 2:
                continue
            row = row_of.get(ui)
            h = rec.state_for_prefix(layer, row, 1) if layer is not None and row is not None else None
            if isinstance(drafter, OracleDrafter):
                drafter.truth = u.ids
            n = u.n_tokens
            ok = rec.top1[u.start : u.end]
            j, s, first = 0, 0, True
            while j < n:
                s += 1  # forward pass that emits token j
                j += 1
                if j >= n:
                    break
                h_cur = rec.state_for_prefix(layer, row, j) if layer is not None and row is not None else None
                d = drafter.draft(u.ids[:j], h, h_cur)
                k = 0
                while k < len(d) and j + k < n and d[k] == u.ids[j + k] and ok[j + k]:
                    k += 1
                drafted_tok += min(len(d), n - j)
                accepted_tok += k
                if first and k == n - 1:
                    exact += 1
                first = False
                j += k
            steps += s
            tokens += n
            words += 1
    return dict(words=words, tokens=tokens, steps=steps, steps_per_token=steps / max(tokens, 1),
                step_reduction=tokens / max(steps, 1), acceptance=accepted_tok / max(drafted_tok, 1),
                whole_word_first_draft=exact / max(words, 1))
