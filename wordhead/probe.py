"""Plan probes: does the hidden state *before* a word already encode the word
beyond its first token?

Setup. Collect occurrences of multi-token words. Group them by their first
token. Within a group every word starts identically, so the model's
next-token prediction cannot tell them apart. A linear probe reads the
pre-word hidden state and must pick the right word *within the group*.

(In byte-level tokenizers almost every Tamil or Hindi word starts with the
same token, a space plus the script's UTF-8 lead bytes, so for those
languages there is essentially one large group and the next-token
distribution says nothing about which word is coming.)

Accuracy is also split by whether the model itself goes on to spell the word
correctly under greedy decoding (``acc_model_spells``): if the pre-word state
holds the plan the model executes, the probe should be far more accurate
there.

Baselines.
* majority  - most frequent word of the group (what a dictionary drafter knows)
* control   - same probe trained on labels permuted within each group
              (keeps label frequencies, destroys the context -> label link)

Train/test splits are by sequence, so no paragraph contributes to both.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from .corpus import SeqRecord
from .models import pick_device


def build_dataset(records: list[SeqRecord], layer: int, min_count: int = 4, min_words_per_group: int = 2,
                  max_classes: int = 1000):
    X, words, firsts, seqs, spelled = [], [], [], [], []
    for si, rec in enumerate(records):
        H = rec.preword_hidden.get(layer)
        if H is None:
            continue
        for row, ui in enumerate(rec.preword_units):
            u = rec.units[ui]
            if u.n_tokens < 2:
                continue
            X.append(H[row])
            words.append(u.ids)
            firsts.append(u.ids[0])
            seqs.append(si)
            spelled.append(bool(rec.top1[u.start + 1 : u.end].all()))
    if not X:
        return None
    cnt = Counter(words)
    groups = defaultdict(set)
    for w in cnt:
        if cnt[w] >= min_count:
            groups[w[0]].add(w)
    keep_words = {w for g in groups.values() if len(g) >= min_words_per_group for w in g}
    keep_words = set(sorted(keep_words, key=lambda w: -cnt[w])[:max_classes])  # most frequent words
    idx = [i for i, w in enumerate(words) if w in keep_words]
    if not idx:
        return None
    vocab = sorted(keep_words)
    wid = {w: i for i, w in enumerate(vocab)}
    return dict(
        X=np.stack([X[i] for i in idx]).astype(np.float32),
        y=np.array([wid[words[i]] for i in idx]),
        first=np.array([firsts[i] for i in idx]),
        seq=np.array([seqs[i] for i in idx]),
        spelled=np.array([spelled[i] for i in idx]),
        vocab=vocab,
        group_of_class=np.array([w[0] for w in vocab]),
    )


def _fit_masked_softmax(Xtr, ytr, gtr, gcls, epochs=40, lr=3e-3, weight_decay=1e-3, batch_size=512, seed=0):
    """Linear softmax trained with the loss restricted to the word's first-token group."""
    torch.manual_seed(seed)
    dev = pick_device()
    X = torch.from_numpy(Xtr).to(dev)
    y = torch.from_numpy(ytr).long().to(dev)
    g = torch.from_numpy(np.asarray(gtr)).long().to(dev)
    gc = torch.from_numpy(np.asarray(gcls)).long().to(dev)
    lin = torch.nn.Linear(X.shape[1], len(gcls)).to(dev)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n).to(dev)
        for i in range(0, n, batch_size):
            b = perm[i : i + batch_size]
            mask = gc[None, :] == g[b][:, None]  # candidates: words of the same group
            logits = lin(X[b]).masked_fill(~mask, -1e9)
            loss = torch.nn.functional.cross_entropy(logits, y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return lin.cpu()


def probe_layer(ds: dict, seed: int = 0, test_size: float = 0.3, epochs: int = 40) -> dict:
    X, y, first, seq = ds["X"], ds["y"], ds["first"], ds["seq"]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(X, y, groups=seq))
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xte = scaler.transform(X[tr]).astype(np.float32), scaler.transform(X[te]).astype(np.float32)
    gcls = ds["group_of_class"]
    test_mask = gcls[None, :] == first[te][:, None]

    def fit_eval(ytr):
        lin = _fit_masked_softmax(Xtr, ytr, first[tr], gcls, epochs=epochs, seed=seed)
        with torch.no_grad():
            scores = lin(torch.from_numpy(Xte)).numpy()
        pred = np.where(test_mask, scores, -np.inf).argmax(1)
        return pred == y[te]

    hit = fit_eval(y[tr])
    acc = float(hit.mean())
    sp = ds.get("spelled")
    acc_spelled = float(hit[sp[te]].mean()) if sp is not None and sp[te].any() else float("nan")
    acc_not_spelled = float(hit[~sp[te]].mean()) if sp is not None and (~sp[te]).any() else float("nan")
    # control: permute labels within each first-token group (training set only)
    rng = np.random.default_rng(seed)
    yperm = y[tr].copy()
    for g in np.unique(first[tr]):
        m = np.where(first[tr] == g)[0]
        yperm[m] = rng.permutation(yperm[m])
    acc_ctrl = float(fit_eval(yperm).mean())
    # majority baseline: most frequent training word inside the group
    maj = {}
    for g in np.unique(first[tr]):
        maj[g] = Counter(y[tr][first[tr] == g]).most_common(1)[0][0]
    maj_pred = np.array([maj.get(g, -1) for g in first[te]])
    acc_maj = float((maj_pred == y[te]).mean())
    # chance: uniform over the words of the group
    sizes = Counter(gcls.tolist())
    chance = float(np.mean([1.0 / sizes[g] for g in first[te]]))
    return dict(acc=acc, acc_control=acc_ctrl, acc_majority=acc_maj, acc_chance=chance, acc_model_spells=acc_spelled,
                acc_model_misspells=acc_not_spelled, n_train=int(len(tr)),
                n_test=int(len(te)), n_classes=int(len(ds["vocab"])), n_groups=int(len(np.unique(gcls))))


def probe_all_layers(records: list[SeqRecord], layers: list[int], seeds=(0, 1, 2), **kw) -> list[dict]:
    out = []
    keys = ("acc", "acc_control", "acc_majority", "acc_chance", "acc_model_spells", "acc_model_misspells")
    for L in layers:
        ds = build_dataset(records, L, **{k: v for k, v in kw.items() if k in ("min_count", "min_words_per_group", "max_classes")})
        if ds is None:
            continue
        res = [probe_layer(ds, seed=s) for s in seeds]
        agg = {k: float(np.nanmean([r[k] for r in res])) for k in keys}
        agg.update({k + "_sd": float(np.nanstd([r[k] for r in res])) for k in keys})
        agg.update(layer=L, n_test=res[0]["n_test"], n_classes=res[0]["n_classes"], n_groups=res[0]["n_groups"])
        out.append(agg)
    return out


def position_probe(records: list[SeqRecord], layer: int, max_prefix: int = 6, min_count: int = 4, seed: int = 0,
                   epochs: int = 40, max_classes: int = 1000) -> list[dict]:
    """Word identity as the word unfolds.

    For j = 1..max_prefix written tokens, a linear probe reads the state that
    emitted token j-1 (pre-word state for j = 1) and predicts the whole word.
    It is compared with what the written prefix alone gives away: the most
    frequent training word with that prefix (a dictionary). Only words that
    still have at least one token left to write are counted at each j.
    """
    cnt = Counter(u.ids for rec in records for ui in rec.preword_units for u in [rec.units[ui]])
    vocab = sorted(sorted((w for w, c in cnt.items() if c >= min_count), key=lambda w: -cnt[w])[:max_classes])
    wid = {w: i for i, w in enumerate(vocab)}
    out = []
    for j in range(1, max_prefix + 1):
        X, y, seq, prefix = [], [], [], []
        for si, rec in enumerate(records):
            for row, ui in enumerate(rec.preword_units):
                u = rec.units[ui]
                if u.ids not in wid or u.n_tokens <= j:
                    continue
                h = rec.state_for_prefix(layer, row, j)
                if h is None:
                    continue
                X.append(h)
                y.append(wid[u.ids])
                seq.append(si)
                prefix.append(u.ids[:j])
        if len(X) < 200:
            break
        X, y, seq = np.stack(X).astype(np.float32), np.array(y), np.array(seq)
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed).split(X, y, groups=seq))
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]).astype(np.float32), scaler.transform(X[te]).astype(np.float32)
        # candidate sets: classes whose word starts with the written prefix
        pref_ids = {}
        pcode = np.array([pref_ids.setdefault(p, len(pref_ids)) for p in prefix])
        class_code = np.array([pref_ids.get(w[:j], -1) for w in vocab])
        lin = _fit_masked_softmax(Xtr, y[tr], pcode[tr], class_code, epochs=epochs, seed=seed)
        with torch.no_grad():
            scores = lin(torch.from_numpy(Xte)).numpy()
        mask = class_code[None, :] == pcode[te][:, None]
        pred = np.where(mask, scores, -np.inf).argmax(1)
        maj = {}
        for g in np.unique(pcode[tr]):
            maj[g] = Counter(y[tr][pcode[tr] == g]).most_common(1)[0][0]
        base = np.array([maj.get(g, -1) for g in pcode[te]])
        out.append(dict(prefix_tokens=j, n_test=int(len(te)), n_classes=len(vocab),
                        acc_state=float((pred == y[te]).mean()), acc_prefix_only=float((base == y[te]).mean()),
                        mean_candidates=float(mask.sum(1).mean())))
    return out
