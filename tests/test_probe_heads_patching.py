import numpy as np
import torch

from wordhead.corpus import SeqRecord, run_corpus
from wordhead.heads import DictionaryDrafter, HeadDrafter, Lexicon, fit_head
from wordhead.patching import _hooks, capture
from wordhead.probe import probe_layer
from wordhead.words import Unit


def test_probe_beats_baselines_when_signal_exists():
    rng = np.random.default_rng(0)
    n, d = 1200, 16
    first = rng.integers(0, 4, n)  # 4 groups
    y_in_group = rng.integers(0, 3, n)  # 3 words per group
    y = first * 3 + y_in_group
    X = rng.normal(size=(n, d)).astype(np.float32)
    X[np.arange(n), y_in_group] += 4.0  # context encodes the word
    ds = dict(X=X, y=y, first=first, seq=np.arange(n) // 4, vocab=list(range(12)),
              group_of_class=np.repeat(np.arange(4), 3))
    r = probe_layer(ds, seed=0)
    assert r["acc"] > 0.8
    assert r["acc_control"] < 0.55 and r["acc_majority"] < 0.55


def test_probe_at_chance_without_signal():
    rng = np.random.default_rng(1)
    n = 1200
    first = rng.integers(0, 4, n)
    y = first * 3 + rng.integers(0, 3, n)
    ds = dict(X=rng.normal(size=(n, 16)).astype(np.float32), y=y, first=first, seq=np.arange(n) // 4,
              vocab=list(range(12)), group_of_class=np.repeat(np.arange(4), 3))
    assert probe_layer(ds, seed=0)["acc"] < 0.5


def _toy_records():
    # two words sharing first token 5: (5, 6, 7) and (5, 8); context decides which
    recs = []
    rng = np.random.default_rng(0)
    for i in range(400):
        w = (5, 6, 7) if i % 2 == 0 else (5, 8)
        ids = [1, 2] + list(w)
        u = Unit(start=2, end=2 + len(w), kind="word", text="a" if i % 2 == 0 else "b", ids=w)
        rec = SeqRecord(ids=ids, units=[Unit(0, 2, "word", "c", (1, 2)), u],
                        logp=np.zeros(len(ids)), entropy=np.zeros(len(ids)), top1=np.ones(len(ids), bool))
        h = rng.normal(size=8).astype(np.float32)
        h[0] += 4 if i % 2 == 0 else -4
        rec.preword_units = [1]
        rec.preword_hidden = {3: h[None].astype(np.float16)}
        recs.append(rec)
    return recs


def test_head_drafter_uses_context_where_dictionary_cannot():
    from wordhead.heads import OracleDrafter, emulated_steps, head_training_data

    recs = _toy_records()
    lex = Lexicon.from_records(recs, size=10)
    assert {(5, 6, 7), (5, 8)} <= set(lex.words)
    X, y = head_training_data(recs, lex, layer=3)
    head = fit_head(X, y, len(lex), epochs=60, lr=1e-2)
    dict_acc = emulated_steps(recs, DictionaryDrafter(lex), layer=None)
    head_acc = emulated_steps(recs, HeadDrafter(lex, head), layer=3)
    oracle = emulated_steps(recs, OracleDrafter(), layer=None)
    assert head_acc["whole_word_first_draft"] > 0.95
    assert dict_acc["whole_word_first_draft"] < 0.6
    # oracle: exactly one forward pass per word; nobody beats it; plain decoding is 1 step per token
    assert oracle["steps"] == oracle["words"]
    assert oracle["steps"] <= head_acc["steps"] <= dict_acc["steps"] <= dict_acc["tokens"]


def test_patching_identity_and_capture(tiny_lm):
    ids = tiny_lm.tokenizer("தமிழ்நாடு இந்தியாவின் தெற்கே", add_special_tokens=False)["input_ids"]
    cap = capture(tiny_lm, ids)
    assert set(cap) == set(range(tiny_lm.n_layers + 1))
    with torch.no_grad():
        base = tiny_lm.model(input_ids=torch.tensor([ids])).logits
    # patching a position with its own activation changes nothing
    p = len(ids) - 3

    def same(t, src=cap[1][p]):
        t = t.clone()
        t[0, p] = src
        return t

    with _hooks(tiny_lm, {1: same}), torch.no_grad():
        patched = tiny_lm.model(input_ids=torch.tensor([ids])).logits
    assert torch.allclose(base, patched, atol=1e-8)
    # patching something else does change downstream logits but not earlier positions
    def other(t, src=cap[1][p] * 0 + 1.0):
        t = t.clone()
        t[0, p] = src
        return t

    with _hooks(tiny_lm, {1: other}), torch.no_grad():
        changed = tiny_lm.model(input_ids=torch.tensor([ids])).logits
    assert torch.allclose(base[0, :p], changed[0, :p], atol=1e-8)
    assert not torch.allclose(base[0, p:], changed[0, p:])


def test_ahead_heads_learn_deterministic_spelling():
    from wordhead.heads import AheadDrafter, ahead_training_data, ahead_vocab, emulated_steps, fit_ahead

    recs = _toy_records()
    # give every word in-word states: the context signal again (so the state identifies the word)
    for rec in recs:
        u = rec.units[1]
        c = u.n_tokens - 2
        rec.inword_offset, rec.inword_count = [0], [c]
        rec.inword_hidden = {3: np.repeat(rec.preword_hidden[3], c, axis=0)}
    vocab = ahead_vocab(recs)
    assert {6, 7, 8} <= set(vocab)
    X, T, Y = ahead_training_data(recs, 3, vocab, K=2)
    emb = torch.randn(10, 4)
    head = fit_ahead(X, T, Y, emb, len(vocab), epochs=40, lr=1e-2, batch_size=64)
    res = emulated_steps(recs, AheadDrafter(head, emb, vocab), layer=3)
    assert res["whole_word_first_draft"] > 0.95  # (5,6,7) and (5,8) both drafted whole, with a STOP after
    assert res["acceptance"] > 0.95


def test_knockout_mask_is_exact_and_local(tiny_lm):
    from wordhead.patching import _cont_logp

    ids = tiny_lm.tokenizer("தமிழ்நாடு இந்தியாவின் தெற்கே அமைந்துள்ள", add_special_tokens=False)["input_ids"]
    s, e = len(ids) - 6, len(ids)
    with torch.no_grad():
        lp = torch.log_softmax(tiny_lm.model(input_ids=torch.tensor([ids])).logits[0, s : e - 1], -1)
    ref = float(lp.gather(-1, torch.tensor(ids[s + 1 : e])[:, None]).sum())
    base, _ = _cont_logp(tiny_lm, ids, s, e, None)
    assert abs(base - ref) < 1e-8  # the explicit 4D causal mask reproduces ordinary attention
    blocked, _ = _cont_logp(tiny_lm, ids, s, e, s - 1)
    assert abs(blocked - base) > 1e-6  # blocking a context position changes the word's probabilities
