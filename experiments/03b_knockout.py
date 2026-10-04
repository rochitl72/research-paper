"""Necessity test: block attention from a word's own positions to the pre-word position.

If the word were chosen at the pre-word position and handed forward, cutting
that single connection should damage the spelling much more than cutting the
connection to another context token.
"""
import argparse

from common import env_info, load_records, results_dir, start_log, write_json

from wordhead.models import load
from wordhead.patching import knockout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--langs", nargs="+", default=["ta", "hi"])
    ap.add_argument("--words", type=int, default=300)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args()
    start_log(f"03b_knockout_{args.model.split('/')[-1]}")
    lm = load(args.model, device=args.device, dtype=args.dtype)
    out = dict(env=env_info(lm), args=vars(args))
    for lang in args.langs:
        res = knockout(lm, load_records(args.model, lang, "eval"), n_words=args.words)
        out[lang] = res
        print(f"[{lang}] {res['n_words']} words the model spells correctly (baseline {res['base_ok']:.0%})")
        for k in ("preword", "preword_minus_1", "random_context"):
            r = res[k]
            print(f"  block {k:>16}: {r['mean_delta_bits']:+.2f} bits on the word's continuation (median {r['median_delta_bits']:+.2f}), "
                  f"still spelled correctly {r['still_spelled']:.0%}")
    write_json(out, results_dir(args.model) / "knockout.json")


if __name__ == "__main__":
    main()
