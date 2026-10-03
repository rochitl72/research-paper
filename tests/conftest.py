import os

import pytest
import torch

TOKENIZER = os.environ.get("WORDHEAD_TEST_TOKENIZER", "Qwen/Qwen3-0.6B-Base")


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER)


@pytest.fixture(scope="session")
def tiny_lm(tokenizer):
    """A randomly initialised 2-layer Qwen3 model with the real tokenizer (fast, deterministic)."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from wordhead.models import Loaded

    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                      tie_word_embeddings=True)
    model = Qwen3ForCausalLM(cfg).to(torch.float64).eval()
    return Loaded(model=model, tokenizer=tokenizer, device=torch.device("cpu"), name="tiny-random-qwen3")
