"""Shared helpers for the experiment scripts (paths, data, caching, logging)."""
from __future__ import annotations

import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
CACHE = Path(os.environ.get("WORDHEAD_CACHE", ROOT / "cache"))
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
LOGS = ROOT / "logs"


def slug(model: str) -> str:
    return model.split("/")[-1]


def load_texts(lang: str, split: str, n: int) -> list[str]:
    """Paragraphs 0-3999 are the 'fit' split, 4000-5999 the held-out 'eval' split.

    The raw files are consecutive Wikipedia articles, so the two splits come
    from different articles.
    """
    with open(DATA / f"wiki_{lang}.jsonl") as f:
        rows = [json.loads(line)["text"] for line in f]
    pool = rows[:4000] if split == "fit" else rows[4000:]
    return pool[:n]


def cache_path(model: str, lang: str, split: str) -> Path:
    p = CACHE / slug(model) / f"{lang}_{split}.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_records(records, model, lang, split):
    with open(cache_path(model, lang, split), "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_records(model, lang, split):
    with open(cache_path(model, lang, split), "rb") as f:
        return pickle.load(f)


def results_dir(model: str) -> Path:
    p = RESULTS / slug(model)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)


def env_info(lm=None) -> dict:
    import torch
    import transformers

    info = dict(python=sys.version.split()[0], torch=torch.__version__, transformers=transformers.__version__,
                platform=platform.platform(), processor=platform.processor(), cpus=os.cpu_count(),
                time=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    if lm is not None:
        info.update(model=lm.name, device=str(lm.device), dtype=str(next(lm.model.parameters()).dtype),
                    n_layers=lm.n_layers, d_model=lm.d_model)
    return info


class Tee:
    """Mirror stdout into a log file under logs/ (kept as the run 'screenshot')."""

    def __init__(self, name: str):
        LOGS.mkdir(parents=True, exist_ok=True)
        self.f = open(LOGS / f"{name}.log", "a")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def start_log(name: str):
    sys.stdout = Tee(name)
    print(f"\n===== {name} | {time.strftime('%Y-%m-%d %H:%M:%S')} | argv: {' '.join(sys.argv[1:])}")
