"""TokenLift: EAGLE head, lossless greedy/sampling paths, head I/O and CLI.

Uses the randomly initialised tiny Qwen3 (float64, so no argmax ties): a random,
untrained head is a *bad* drafter, which is exactly what a losslessness test needs.
"""
import subprocess
import sys

import pytest
import torch

from tokenlift import TokenLift
from wordhead.eagle import WordEagleHead

PROMPTS = ["தமிழ்நாடு இந்தியாவின் தெற்கே அமைந்துள்ள", "भारत की राजधानी", "The capital of France is"]
N_VOCAB = 50


@pytest.fixture()
def spec(tiny_lm):
    s = TokenLift(tiny_lm)
    torch.manual_seed(1)
    head = WordEagleHead(tiny_lm.d_model, s.emb.shape[1], N_VOCAB, hidden=32)
    vocab = list(range(100, 100 + N_VOCAB))
    s._attach("ta", head.eval(), vocab, max_draft=4)
    return s


def test_head_shapes():
    head = WordEagleHead(64, 64, N_VOCAB, hidden=32)
    f, logits = head(torch.randn(5, 64), torch.randn(5, 64))
    assert f.shape == (5, 64)
    assert logits.shape == (5, N_VOCAB + 2)
    assert head.other == N_VOCAB and head.stop == N_VOCAB + 1


def test_greedy_is_lossless(spec):
    for p in PROMPTS:
        fast = spec.generate(p, max_new=24, lang="ta", accelerate=True)
        plain = spec.generate(p, max_new=24, lang="ta", accelerate=False)
        assert fast["tokens"] == plain["tokens"]


def test_sampling_at_zero_temperature_equals_greedy(spec):
    for p in PROMPTS:
        plain = spec.generate(p, max_new=24, accelerate=False)
        samp = spec.sample(p, max_new=24, temperature=1e-5, lang="ta", seed=3)
        assert samp["tokens"] == plain["tokens"]


def test_sampling_is_seeded_and_reports_stats(spec):
    a = spec.sample(PROMPTS[0], max_new=20, temperature=1.0, seed=7)
    b = spec.sample(PROMPTS[0], max_new=20, temperature=1.0, seed=7)
    assert a["tokens"] == b["tokens"]
    assert a["tokens_per_forward"] > 0
    assert 0.0 <= a["acceptance"] <= 1.0
    assert len(a["tokens"]) <= 20


def test_save_load_roundtrip(spec, tmp_path):
    path = tmp_path / "h.pt"
    spec.save_head("ta", path)
    fresh = TokenLift(spec.lm).load_head("ta", path)
    for k, v in spec.heads["ta"].state_dict().items():
        assert torch.equal(v, fresh.heads["ta"].state_dict()[k])
    assert fresh.vocabs["ta"] == spec.vocabs["ta"]


def test_load_bare_state_dict_needs_matching_vocab(spec, tmp_path):
    path = tmp_path / "bare.pt"
    torch.save(spec.heads["ta"].state_dict(), path)
    ok = TokenLift(spec.lm).load_head("ta", path, vocab=spec.vocabs["ta"])
    assert "ta" in ok.heads
    with pytest.raises(AssertionError):
        TokenLift(spec.lm).load_head("ta", path, vocab=spec.vocabs["ta"][:-1])


def test_generate_without_head_raises(tiny_lm):
    with pytest.raises(RuntimeError):
        TokenLift(tiny_lm).generate("hello", max_new=4)


def test_cli_help():
    r = subprocess.run([sys.executable, "-m", "tokenlift.cli", "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "report" in r.stdout and "gen" in r.stdout
