"""Multi-seed robustness for the knockout (necessity) result -- the cheap,
model-light part. For each seed we resample the words and re-run knockout;
we report the spread of the pre-word cost and still-spelled rate across seeds.
Writes knockout_seeds.json.

    python experiments/09_knockout_seeds.py --model Qwen/Qwen3-0.6B-Base --seeds 0 1 2
"""
import argparse

import numpy as np

from common import env_info, load_records, results_dir, start_log, write_json

from wordhead.models import load
from wordhead.patching import knockout


def summ(xs):
    a = np.array(xs, float)
    return dict(mean=float(a.mean()), sd=float(a.std()), values=[round(float(x), 3) for x in xs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--langs", nargs="+", default=["ta", "hi"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--words", type=int, default=200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args()
    start_log(f"09_knockout_seeds_{args.model.split('/')[-1]}")
    lm = load(args.model, device=args.device, dtype=args.dtype)
    out = dict(env=env_info(lm), args=vars(args), by_lang={})
    for lang in args.langs:
        recs = load_records(args.model, lang, "fit") + load_records(args.model, lang, "eval")
        pre_bits, rnd_bits, spelled = [], [], []
        for s in args.seeds:
            ko = knockout(lm, recs, n_words=args.words, seed=s, progress=False)
            pre_bits.append(ko["preword"]["mean_delta_bits"])
            rnd_bits.append(ko["random_context"]["mean_delta_bits"])
            spelled.append(ko["preword"]["still_spelled"])
            print(f"[{lang} seed {s}] pre-word {pre_bits[-1]:+.2f} bits (random {rnd_bits[-1]:+.2f}), "
                  f"still-spelled {spelled[-1]:.2f}")
        out["by_lang"][lang] = dict(n_seeds=len(args.seeds), words=args.words,
                                    preword_bits=summ(pre_bits), random_bits=summ(rnd_bits),
                                    still_spelled=summ(spelled))
    write_json(out, results_dir(args.model) / "knockout_seeds.json")


if __name__ == "__main__":
    main()
