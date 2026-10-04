"""Step 0: one teacher-forced pass per (model, language, split); cache records.

Example (M1 Pro):
    python experiments/00_corpus_pass.py --model Qwen/Qwen3-1.7B-Base --langs ta hi en --n-fit 2000 --n-eval 1000
"""
import argparse
import time

from common import env_info, load_texts, save_records, start_log, write_json, results_dir

from wordhead.corpus import run_corpus
from wordhead.models import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--langs", nargs="+", default=["ta", "hi", "en"])
    ap.add_argument("--n-fit", type=int, default=1000)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--inword-positions", type=int, default=6)
    ap.add_argument("--layer-step", type=int, default=7, help="store pre-word hidden states every k layers")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args()
    start_log(f"00_corpus_pass_{args.model.split('/')[-1]}")
    lm = load(args.model, device=args.device, dtype=args.dtype)
    layers = sorted(set(list(range(0, lm.n_layers + 1, args.layer_step)) + [lm.n_layers]))
    meta = dict(env=env_info(lm), args=vars(args), hidden_layers=layers, runs=[])
    print(meta["env"])
    for lang in args.langs:
        for split, n in (("fit", args.n_fit), ("eval", args.n_eval)):
            texts = load_texts(lang, split, n)
            t0 = time.time()
            print(f"[{lang}/{split}] {len(texts)} paragraphs")
            recs = run_corpus(lm, texts, max_len=args.max_len, batch_size=args.batch_size, hidden_layers=layers,
                              inword_layers=[lm.n_layers], inword_positions=args.inword_positions)
            dt = time.time() - t0
            n_tok = sum(len(r.ids) for r in recs)
            print(f"[{lang}/{split}] {n_tok} tokens in {dt:.1f}s ({n_tok / dt:.0f} tok/s)")
            save_records(recs, args.model, lang, split)
            meta["runs"].append(dict(lang=lang, split=split, paragraphs=len(texts), tokens=n_tok, seconds=dt))
    write_json(meta, results_dir(args.model) / "corpus_pass_meta.json")


if __name__ == "__main__":
    main()
