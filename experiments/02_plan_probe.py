"""Kill test 2a: linear probes for the word plan at the pre-word position.

Within groups of words sharing a first token, can a probe on the hidden state
*before* the word pick the right word? Compared with the group's majority
word (dictionary knowledge) and a label-permutation control.
"""
import argparse

from common import load_records, results_dir, start_log, write_json

from wordhead.probe import position_probe, probe_all_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--langs", nargs="+", default=["ta", "hi", "en"])
    ap.add_argument("--min-count", type=int, default=4)
    args = ap.parse_args()
    start_log(f"02_plan_probe_{args.model.split('/')[-1]}")
    out = {}
    for lang in args.langs:
        recs = load_records(args.model, lang, "fit") + load_records(args.model, lang, "eval")
        layers = sorted(recs[0].preword_hidden)
        res = probe_all_layers(recs, layers, min_count=args.min_count)
        out[lang] = dict(by_layer=res)
        if res:
            print(f"\n[{lang}] classes={res[0]['n_classes']} groups={res[0]['n_groups']} test n={res[0]['n_test']}")
        print(f"  {'layer':>5} {'probe':>7} {'majority':>9} {'control':>8} {'chance':>7} | {'model spells':>12} {'misspells':>9}")
        for r in res:
            print(f"  {r['layer']:>5} {r['acc']:>7.3f} {r['acc_majority']:>9.3f} {r['acc_control']:>8.3f} {r['acc_chance']:>7.3f} | "
                  f"{r['acc_model_spells']:>12.3f} {r['acc_model_misspells']:>9.3f}")
        last = max(recs[0].inword_hidden) if recs[0].inword_hidden else None
        if last is not None:
            pos = position_probe(recs, last, min_count=args.min_count)
            out[lang]["by_position"] = pos
            print(f"  word identity as the word unfolds (layer {last}): tokens written -> state probe vs prefix-only dictionary")
            for r in pos:
                print(f"    {r['prefix_tokens']} tokens: state {r['acc_state']:.3f} | prefix only {r['acc_prefix_only']:.3f} "
                      f"(n={r['n_test']}, {r['mean_candidates']:.0f} candidate words)")
    write_json(out, results_dir(args.model) / "plan_probe.json")


if __name__ == "__main__":
    main()
