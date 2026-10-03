"""Model loading that works the same on Apple Silicon (MPS), CUDA and CPU."""
from __future__ import annotations

from dataclasses import dataclass

import torch


def pick_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pick_dtype(device: torch.device, dtype: str = "auto") -> torch.dtype:
    if dtype != "auto":
        return getattr(torch, dtype)
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


@dataclass
class Loaded:
    model: torch.nn.Module
    tokenizer: object
    device: torch.device
    name: str

    @property
    def n_layers(self) -> int:
        return self.model.config.num_hidden_layers

    @property
    def d_model(self) -> int:
        return self.model.config.hidden_size


def load(name: str, device: str = "auto", dtype: str = "auto") -> Loaded:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = pick_device(device)
    dt = pick_dtype(dev, dtype)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dt)
    model.to(dev).eval()
    return Loaded(model=model, tokenizer=tok, device=dev, name=name)


def decoder_layers(model: torch.nn.Module):
    """Return the list of transformer blocks for common HF decoder architectures."""
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            return obj
    raise ValueError("Unsupported architecture: cannot find decoder layers")
