"""Plan probes: does the hidden state *before* a word already encode the word
beyond its first token?

Setup. Collect occurrences of multi-token words. Group them by their first
token. Within a group every word starts identically, so the model's
next-token prediction cannot tell them apart. A linear probe reads the
pre-word hidden state and must pick the right word *within the group*.

Baselines.
* majority  - most frequent word of the group (what a dictionary drafter knows)
* control   - same probe trained on labels permuted within each group
              (keeps label frequencies, destroys the context -> label link)

Train/test splits are by sequence, so no paragraph contributes to both.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from .corpus import SeqRecord


def build_dataset(records: list[SeqRecord], layer: int, min_count: int = 4, min_words_per_group: int = 2):
    X, words, firsts, seqs = [], [], [], []
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
    if not X:
        return None
    cnt = Counter(words)
    groups = defaultdict(set)
    for w in cnt:
        if cnt[w] >= min_count:
            groups[w[0]].add(w)
    keep_words = {w for g in groups.values() if len(g) >= min_words_per_group for w in g}
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
        vocab=vocab,
        group_of_class=np.array([w[0] for w in vocab]),
    )


def _within_group_predict(scores: np.ndarray, first: np.ndarray, group_of_class: np.ndarray) -> np.ndarray:
    masked = np.where(group_of_class[None, :] == first[:, None], scores, -np.inf)
    return masked.argmax(1)


def probe_layer(ds: dict, seed: int = 0, C: float = 0.1, test_size: float = 0.3) -> dict:
    X, y, first, seq = ds["X"], ds["y"], ds["first"], ds["seq"]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(X, y, groups=seq))
    # classes seen in training only
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
    gcls = ds["group_of_class"]

    def fit_eval(ytr):
        clf = LogisticRegression(C=C, max_iter=300)
        clf.fit(Xtr, ytr)
        scores = np.full((len(te), len(gcls)), -np.inf)
        scores[:, clf.classes_] = clf.decision_function(Xte) if len(clf.classes_) > 2 else np.stack(
            [-clf.decision_function(Xte), clf.decision_function(Xte)], 1)
        pred = _within_group_predict(scores, first[te], gcls)
        return float((pred == y[te]).mean())

    acc = fit_eval(y[tr])
    # control: permute labels within each first-token group (training set only)
    rng = np.random.default_rng(seed)
    yperm = y[tr].copy()
    for g in np.unique(first[tr]):
        m = np.where(first[tr] == g)[0]
        yperm[m] = rng.permutation(yperm[m])
    acc_ctrl = fit_eval(yperm)
    # majority baseline: most frequent training word inside the group
    maj = {}
    for g in np.unique(first[tr]):
        maj[g] = Counter(y[tr][first[tr] == g]).most_common(1)[0][0]
    maj_pred = np.array([maj.get(g, -1) for g in first[te]])
    acc_maj = float((maj_pred == y[te]).mean())
    return dict(acc=acc, acc_control=acc_ctrl, acc_majority=acc_maj, n_train=int(len(tr)), n_test=int(len(te)),
                n_classes=int(len(ds["vocab"])), n_groups=int(len(np.unique(gcls))))


def probe_all_layers(records: list[SeqRecord], layers: list[int], seeds=(0, 1, 2), **kw) -> list[dict]:
    out = []
    for L in layers:
        ds = build_dataset(records, L, **{k: v for k, v in kw.items() if k in ("min_count", "min_words_per_group")})
        if ds is None:
            continue
        res = [probe_layer(ds, seed=s) for s in seeds]
        agg = {k: float(np.mean([r[k] for r in res])) for k in ("acc", "acc_control", "acc_majority")}
        agg.update({k + "_sd": float(np.std([r[k] for r in res])) for k in ("acc", "acc_control", "acc_majority")})
        agg.update(layer=L, n_test=res[0]["n_test"], n_classes=res[0]["n_classes"], n_groups=res[0]["n_groups"])
        out.append(agg)
    return out
