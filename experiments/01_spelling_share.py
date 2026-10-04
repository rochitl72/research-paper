"""Kill test 1: spelling share per language (forward passes only).

Reads cached records from 00_corpus_pass.py.
"""
import argparse

from common import load_records, results_dir, start_log, write_json

from wordhead.spelling import bits_by_position, by_length, frequency_matched_by_length, spelling_summary, word_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--langs", nargs="+", default=["ta", "hi", "en"])
    args = ap.parse_args()
    start_log(f"01_spelling_share_{args.model.split('/')[-1]}")
    out = {}
    for lang in args.langs:
        recs = load_records(args.model, lang, "fit") + load_records(args.model, lang, "eval")
        rows = word_table(recs)
        s = spelling_summary(rows)
        s["by_length"] = by_length(rows)
        s["freq_matched"] = frequency_matched_by_length(rows)
        s["by_position"] = bits_by_position(rows)
        out[lang] = s
        print(f"\n[{lang}] words={s['n_words']}  tokens/word={s['tokens_per_word']:.2f}  "
              f"multi-token words={s['share_words_multitoken']:.1%}")
        print(f"  continuation tokens = {s['cont_share_of_word_tokens']:.1%} of word tokens")
        for tau in (0.5, 0.9, 0.99):
            print(f"  spelling share @p>={tau}: {s[f'spelling_share@{tau}']:.1%} of word tokens "
                  f"({s[f'cont_deterministic@{tau}']:.1%} of continuations)")
        print(f"  greedy predicts continuation: {s['cont_greedy_correct']:.1%}")
        print(f"  bits/word: first token {s['bits_first_per_word']:.2f}, continuation {s['bits_cont_per_word']:.2f} "
              f"-> {s['share_bits_in_first_token']:.1%} of word information is in the first token")
        print("  boundary mix:", {k: f"{v:.1%}" for k, v in s["boundary_share_of_cont"].items()})
        print("  deterministic@0.9 by boundary:", {k: f"{v:.1%}" for k, v in s["boundary_deterministic@0.9"].items()})
        print("  bits by token position in word:", " ".join(f"{d['bits']:.2f}" for d in s["by_position"][:12]))
    write_json(out, results_dir(args.model) / "spelling_share.json")


if __name__ == "__main__":
    main()
