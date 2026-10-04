"""Kill test 2b: causal activation patching between natural contexts.

Patch the pre-word residual of context A (model writes word X) into context B
(model writes word Y, same first token) and measure whether the model now
spells X's later tokens. Positive control: patch the first-token position.
"""
import argparse

from common import env_info, load_records, results_dir, start_log, write_json

from wordhead.models import load
from wordhead.patching import mine_pairs, sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--langs", nargs="+", default=["ta"])
    ap.add_argument("--pairs", type=int, default=60)
    ap.add_argument("--layer-step", type=int, default=2)
    ap.add_argument("--ctx-len", type=int, default=64)
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args()
    start_log(f"03_patching_{args.model.split('/')[-1]}")
    lm = load(args.model, device=args.device, dtype=args.dtype)
    layers = sorted(set(range(0, lm.n_layers + 1, args.layer_step)) | {lm.n_layers})
    out = dict(env=env_info(lm), args=vars(args))
    for lang in args.langs:
        recs = load_records(args.model, lang, "eval") + load_records(args.model, lang, "fit")
        pairs = mine_pairs(recs, min_conf=args.min_conf, ctx_len=args.ctx_len, max_pairs=args.pairs)
        print(f"[{lang}] {len(pairs)} pairs")
        res = {}
        for where in ("preword", "first_token"):
            r = sweep(lm, pairs, layers, where=where)
            res[where] = r
            print(f"  {where}: layer -> normalized effect (median) | exact transfer of X's continuation")
            for L, e, t in zip(r["layers"], r["effect_median"], r["transfer_rate"]):
                print(f"    L{L:>2}: {e:+.2f} | {t:.0%}")
        res["examples"] = [dict(A_word=lm.tokenizer.decode(p["A"]["word"]), B_word=lm.tokenizer.decode(p["B"]["word"]),
                                A_ctx=lm.tokenizer.decode(p["A"]["ctx"][-24:]), B_ctx=lm.tokenizer.decode(p["B"]["ctx"][-24:]))
                           for p in pairs[:10]]
        out[lang] = res
    write_json(out, results_dir(args.model) / "patching.json")


if __name__ == "__main__":
    main()
