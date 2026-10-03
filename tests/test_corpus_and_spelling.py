import math

import numpy as np
import torch

from wordhead.corpus import run_corpus
from wordhead.spelling import spelling_summary, word_table

TEXTS = ["குழந்தைகள் வீட்டிலிருந்து பள்ளிக்கு நடந்து சென்றனர்.", "The children walked to school.", "बच्चे घर से स्कूल तक पैदल गए।"]


def test_teacher_forcing_matches_manual_chain_rule(tiny_lm):
    recs = run_corpus(tiny_lm, TEXTS, batch_size=3, progress=False)
    for text, rec in zip(TEXTS, recs):
        ids = tiny_lm.tokenizer(text, add_special_tokens=False)["input_ids"]
        assert rec.ids == ids
        with torch.no_grad():
            lp = torch.log_softmax(tiny_lm.model(input_ids=torch.tensor([ids])).logits[0].double(), -1)
        manual = [float(lp[j - 1, ids[j]]) for j in range(1, len(ids))]
        np.testing.assert_allclose(rec.logp[1:], manual, rtol=1e-4, atol=1e-4)
        assert math.isnan(rec.logp[0])


def test_padding_does_not_change_stats(tiny_lm):
    a = run_corpus(tiny_lm, TEXTS, batch_size=3, progress=False)
    b = run_corpus(tiny_lm, TEXTS, batch_size=1, progress=False)
    for ra, rb in zip(a, b):
        np.testing.assert_allclose(ra.logp[1:], rb.logp[1:], atol=1e-4)


def test_hidden_states_collected_at_preword_positions(tiny_lm):
    recs = run_corpus(tiny_lm, TEXTS, batch_size=2, hidden_layers=[0, 2], progress=False)
    for rec in recs:
        n = len(rec.preword_units)
        assert rec.preword_hidden[0].shape == (n, 64)
        assert rec.preword_hidden[2].shape == (n, 64)
        for ui in rec.preword_units:
            assert rec.units[ui].kind == "word" and rec.units[ui].start >= 1
    # layer-0 state at the pre-word position is just the embedding of that token
    rec = recs[0]
    emb = tiny_lm.model.get_input_embeddings().weight
    for row, ui in enumerate(rec.preword_units):
        tid = rec.ids[rec.units[ui].start - 1]
        np.testing.assert_allclose(rec.preword_hidden[0][row], emb[tid].detach().numpy(), atol=1e-3)


def test_spelling_summary_consistency(tiny_lm):
    recs = run_corpus(tiny_lm, TEXTS * 2, batch_size=3, progress=False)
    rows = word_table(recs)
    s = spelling_summary(rows)
    assert s["n_word_tokens"] == s["n_words"] + s["n_cont_tokens"]
    assert 0 <= s["spelling_share@0.9"] <= s["cont_share_of_word_tokens"] <= 1
    assert sum(s["boundary_counts"].values()) == s["n_cont_tokens"]
    assert s["spelling_share@0.99"] <= s["spelling_share@0.9"] <= s["spelling_share@0.5"]
