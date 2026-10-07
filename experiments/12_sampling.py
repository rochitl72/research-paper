"""Benchmark lossless speculative *sampling* (temperature>0) with the trained
word-scoped EAGLE head, across several prompts per language. Writes
results/<model>/sampling.json. The accept/reject rule preserves the target's
sampling distribution by construction (Leviathan 2023; Chen 2023), so this
measures speed, not quality."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from indicspec import IndicSpec
from experiments.common import load_texts, cache_path, results_dir, write_json, slug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--langs", nargs="+", default=["ta", "hi"])
    ap.add_argument("--temps", nargs="+", type=float, default=[0.7, 1.0])
    ap.add_argument("--prompts", type=int, default=12)
    ap.add_argument("--prompt-tokens", type=int, default=32)
    ap.add_argument("--new-tokens", type=int, default=64)
    a = ap.parse_args()

    spec = IndicSpec.from_pretrained(a.model)
    out = {"args": vars(a), "by_lang": {}}
    for lang in a.langs:
        head_pt = cache_path(a.model, lang, "eagle_head").with_suffix(".pt")
        if not head_pt.exists():
            print(f"[skip {lang}] no head at {head_pt}"); continue
        spec.load_head(lang, str(head_pt))
        texts = load_texts(lang, "eval", a.prompts * 3)
        tok = spec.lm.tokenizer
        prompts = []
        for t in texts:
            ids = tok(t, add_special_tokens=False)["input_ids"][: a.prompt_tokens]
            if len(ids) >= a.prompt_tokens:
                prompts.append(ids)
            if len(prompts) >= a.prompts:
                break
        rec = {}
        for T in a.temps:
            tpf, acc, sec = [], [], []
            for i, ids in enumerate(prompts):
                r = spec.sample(ids, max_new=a.new_tokens, temperature=T, lang=lang, seed=i)
                tpf.append(r["tokens_per_forward"]); acc.append(r["acceptance"])
            rec[f"T={T}"] = dict(tokens_per_forward=float(np.mean(tpf)),
                                 tok_per_fwd_std=float(np.std(tpf)),
                                 acceptance=float(np.mean(acc)), n_prompts=len(prompts))
            print(f"  {lang} T={T}: {np.mean(tpf):.2f} tok/fwd (+/-{np.std(tpf):.2f}), accept {np.mean(acc):.1%}")
        out["by_lang"][lang] = rec
    write_json(out, results_dir(a.model) / "sampling.json")
    print("wrote", results_dir(a.model) / "sampling.json")


if __name__ == "__main__":
    main()
