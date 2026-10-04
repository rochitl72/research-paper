"""The decoder must be lossless: identical tokens to plain greedy decoding."""
import random

import torch

from wordhead.specdec import generate

PROMPTS = ["தமிழ்நாடு இந்தியாவின் தெற்கே அமைந்துள்ள", "The capital of France is", "भारत की राजधानी"]


def hf_greedy(lm, ids, n):
    with torch.no_grad():
        out = lm.model.generate(torch.tensor([ids]), max_new_tokens=n, do_sample=False, min_new_tokens=n,
                                pad_token_id=lm.tokenizer.eos_token_id)
    return out[0, len(ids):].tolist()


class RandomDrafter:
    name = "dictionary"

    def __init__(self, vocab, seed=0):
        self.rng = random.Random(seed)
        self.vocab = vocab

    def draft(self, prefix, h=None, h_cur=None):
        return [self.rng.randrange(self.vocab) for _ in range(self.rng.randint(0, 4))]


class ReplayDrafter:
    """Proposes the next k tokens of the known greedy continuation (optionally corrupted)."""
    name = "dictionary"

    def __init__(self, ref, k=3, corrupt_every=0):
        self.ref, self.k, self.corrupt_every, self.calls, self.n = ref, k, corrupt_every, 0, 0

    def observe(self, committed):
        self.n = len(committed)

    def draft(self, prefix, h=None, h_cur=None):
        self.calls += 1
        d = list(self.ref[self.n : self.n + self.k])
        if self.corrupt_every and self.calls % self.corrupt_every == 0 and d:
            d[-1] = (d[-1] + 1) % 1000
        return d


def test_plain_loop_equals_hf_generate(tiny_lm):
    for p in PROMPTS:
        ids = tiny_lm.tokenizer(p, add_special_tokens=False)["input_ids"]
        r = generate(tiny_lm, ids, 24, drafter=None, stop_ids=set())
        assert r.tokens == hf_greedy(tiny_lm, ids, 24)
        assert r.n_forward == 1 + 24


def test_random_drafts_are_lossless(tiny_lm):
    for seed, p in enumerate(PROMPTS):
        ids = tiny_lm.tokenizer(p, add_special_tokens=False)["input_ids"]
        ref = generate(tiny_lm, ids, 30, drafter=None, stop_ids=set()).tokens
        r = generate(tiny_lm, ids, 30, drafter=RandomDrafter(len(tiny_lm.tokenizer), seed), stop_ids=set())
        assert r.tokens == ref


def test_correct_drafts_save_forward_passes_and_stay_lossless(tiny_lm):
    for p in PROMPTS:
        ids = tiny_lm.tokenizer(p, add_special_tokens=False)["input_ids"]
        ref = generate(tiny_lm, ids, 30, drafter=None, stop_ids=set()).tokens
        perfect = generate(tiny_lm, ids, 30, drafter=ReplayDrafter(ref, k=3), stop_ids=set())
        assert perfect.tokens == ref
        # each forward commits 1 + 3 tokens when every draft is right
        assert perfect.n_forward <= 1 + -(-30 // 4) + 1
        assert all(a == l for a, l in zip(perfect.accepted, perfect.draft_lengths))
        partial = generate(tiny_lm, ids, 30, drafter=ReplayDrafter(ref, k=3, corrupt_every=2), stop_ids=set())
        assert partial.tokens == ref
        assert perfect.n_forward < partial.n_forward < 1 + 30


def test_drafter_receives_the_state_that_emitted_the_latest_token(tiny_lm):
    """The decoder must hand drafters exactly the states the corpus pass caches."""
    ids = tiny_lm.tokenizer(PROMPTS[0], add_special_tokens=False)["input_ids"]
    seen = []

    class Spy:
        name = "head"

        def observe(self, committed):
            self.n = len(committed)

        def draft(self, prefix, h=None, h_cur=None):
            seen.append((self.n, h_cur.clone()))
            return []

    r = generate(tiny_lm, ids, 10, drafter=Spy(), layer=2, stop_ids=set())
    full = ids + r.tokens
    with torch.no_grad():
        hs = tiny_lm.model(input_ids=torch.tensor([full]), output_hidden_states=True).hidden_states[2][0]
    for n, h in seen:
        # n tokens committed (including the latest one) -> state at the position just before the latest token
        assert torch.allclose(h, hs[len(ids) + n - 2], atol=1e-6)
