"""Integration checks on a real pretrained model (run with: pytest -m slow)."""
import os

import pytest

pytestmark = pytest.mark.slow

MODEL = os.environ.get("WORDHEAD_TEST_MODEL", "Qwen/Qwen3-0.6B-Base")


@pytest.fixture(scope="module")
def lm():
    from wordhead.models import load
    return load(MODEL, device="cpu", dtype="float32")


def test_lossless_on_real_model_with_dictionary_drafts(lm):
    from wordhead.corpus import run_corpus
    from wordhead.heads import DictionaryDrafter, Lexicon
    from wordhead.specdec import generate

    texts = ["தமிழ்நாடு இந்தியாவின் தெற்கே அமைந்துள்ள ஒரு மாநிலம். அதன் தலைநகரம் சென்னை ஆகும்."] * 3
    recs = run_corpus(lm, texts, progress=False)
    lex = Lexicon.from_records(recs, size=100, min_count=1)
    prompt = lm.tokenizer("தமிழ்நாடு இந்தியாவின்", add_special_tokens=False)["input_ids"]
    plain = generate(lm, prompt, 40, drafter=None, stop_ids=set())
    fast = generate(lm, prompt, 40, drafter=DictionaryDrafter(lex), stop_ids=set())
    assert fast.tokens == plain.tokens
    assert fast.n_forward <= plain.n_forward
