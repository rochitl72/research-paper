"""TokenLift: lossless word-level speculative decoding for Indic-script LLMs.

A drop-in accelerator for generation in heavily fragmented scripts (Tamil,
Hindi, ...). It trains a small word-scoped autoregressive draft head on the
target model's own features, then drafts whole words and verifies them in one
forward pass -- producing output that matches the target model (greedy: by
construction; sampling: distribution-preserving).

    from tokenlift import TokenLift
    spec = TokenLift.from_pretrained("Qwen/Qwen3-0.6B-Base")
    spec.fit(["ta"])                      # or spec.load_head("ta", "head.pt")
    print(spec.generate("...prompt...", max_new=64))          # lossless greedy
    print(spec.sample("...prompt...", max_new=64, temperature=0.8))  # lossless sampling
"""
from .core import TokenLift

__all__ = ["TokenLift"]
__version__ = "0.1.0"
