"""Seed robustness for patching and knockout: does the random selection of
pairs / words change the conclusion? Re-runs each with several seeds and
reports the spread of the key statistics. Writes seed_robustness.json.

    python experiments/08_seed_robustness.py --model Qwen/Qwen3-0.6B-Base --seeds 0 1 2
"""
import argparse

import numpy as np

from common import env_info, load_records, results_dir, start_log, write_json

from wordhead.models import load
from wordhead.patching import knockout, mine_pairs, sweep


def summ(xs):
    a = np.array(xs, float)
    return dict(mean=float(a.mean()), sd=float(a.std()), min=float(a.min()), max=float(a.max()), values=[float(x) for x in xs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--langs", nargs="+", default=["ta", "hi"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--pairs", type=int, default=80)
    ap.add_argument("--words", type=int, default=200)
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args()
    start_log(f"08_seed_robustness_{args.model.split('/')[-1]}")
    lm = load(args.model, device=args.device, dtype=args.dtype)
    layers = list(range(0, lm.n_layers + 1, 2))
    out = dict(env=env_info(lm), args=vars(args), by_lang={})
    for lang in args.langs:
        recs = load_records(args.model, lang, "fit") + load_records(args.model, lang, "eval")
        ft_transfer_last, pre_eff_best, ko_bits, ko_spelled, ko_rand_bits = [], [], [], [], []
        for s in args.seeds:
            pairs = mine_pairs(recs, min_conf=args.min_conf, max_pairs=args.pairs, seed=s)
            pre = sweep(lm, pairs, layers, where="preword", progress=False)
            ft = sweep(lm, pairs, layers, where="first_token", progress=False)
            ft_transfer_last.append(ft["transfer_rate"][-1])
            pre_eff_best.append(float(np.nanmax(pre["effect_median"])))
            ko = knockout(lm, recs, n_words=args.words, seed=s, progress=False)
            ko_bits.append(ko["preword"]["mean_delta_bits"])
            ko_rand_bits.append(ko["random_context"]["mean_delta_bits"])
            ko_spelled.append(ko["preword"]["still_spelled"])
            print(f"[{lang} seed {s}] first-token transfer(last) {ft['transfer_rate'][-1]:.2f}, "
                  f"pre-word max median effect {pre_eff_best[-1]:.3f}, "
                  f"knockout pre-word {ko_bits[-1]:+.2f} bits (random {ko_rand_bits[-1]:+.2f}), "
                  f"still-spelled {ko_spelled[-1]:.2f}")
        out["by_lang"][lang] = dict(
            n_seeds=len(args.seeds), pairs=args.pairs, words=args.words,
            first_token_transfer_last=summ(ft_transfer_last),
            preword_max_median_effect=summ(pre_eff_best),
            knockout_preword_bits=summ(ko_bits),
            knockout_random_bits=summ(ko_rand_bits),
            knockout_still_spelled=summ(ko_spelled),
        )
    write_json(out, results_dir(args.model) / "seed_robustness.json")


if __name__ == "__main__":
    main()
