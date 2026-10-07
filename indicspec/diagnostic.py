"""Headroom diagnostic: given a model and some text, estimate how much IndicSpec
can accelerate generation, per language, from the tokenizer's fragmentation and
the model's own continuation predictability."""
from __future__ import annotations

from wordhead.corpus import run_corpus
from wordhead.spelling import spelling_summary, word_table


def headroom(lm, texts, n=120, max_len=256, batch_size=4):
    """Return a per-call report. `texts` is a list of strings in one language."""
    recs = run_corpus(lm, texts[:n], max_len=max_len, batch_size=batch_size, hidden_layers=[], progress=False)
    s = spelling_summary(word_table(recs))
    tpw = s["tokens_per_word"]
    g = s["cont_greedy_correct"]              # fraction of continuation tokens the model gets greedily right
    # expected forward passes per word if each continuation needs a new pass w.p. (1-g)
    exp_passes = 1 + (1 - g) * max(tpw - 1, 0)
    est_tpf = tpw / max(exp_passes, 1e-6)      # estimated best tokens per forward pass
    return dict(
        tokens_per_word=round(tpw, 2),
        first_token_info_share=round(s["share_bits_in_first_token"], 3),
        spelling_share_0_9=round(s["spelling_share@0.9"], 3),
        greedy_correct_continuations=round(g, 3),
        estimated_tokens_per_forward=round(est_tpf, 2),
        headroom="high" if est_tpf > 1.6 else ("medium" if est_tpf > 1.2 else "low"),
    )


def report(lm, texts_by_lang, **kw):
    out = {}
    for lang, texts in texts_by_lang.items():
        out[lang] = headroom(lm, texts, **kw)
    return out
