"""Train and evaluate the word-scoped EAGLE head vs the current best drafter.

    python experiments/11_eagle.py --model Qwen/Qwen3-0.6B-Base --langs ta hi
"""
import argparse
import time

import torch

from common import cache_path, env_info, load_records, load_texts, results_dir, start_log, write_json

from wordhead.eagle import (WordEagleDrafter, ahead_vocab, eagle_seq_data, fit_word_eagle_rollout)
from wordhead.heads import (AheadDrafter, DictionaryDrafter, HeadDrafter, Lexicon, OracleDrafter,
                            ahead_training_data, emulated_steps, fit_ahead, fit_head, head_training_data)
from wordhead.heads import ahead_vocab as ahead_vocab_heads
from wordhead.models import load
from wordhead.specdec import generate
from wordhead.words import Segmenter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--langs", nargs="+", default=["ta", "hi"])
    ap.add_argument("--vocab", type=int, default=4000)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--max-draft", type=int, default=10)
    ap.add_argument("--prompts", type=int, default=24)
    ap.add_argument("--prompt-tokens", type=int, default=32)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args()
    start_log(f"11_eagle_{args.model.split('/')[-1]}")
    lm = load(args.model, device=args.device, dtype=args.dtype)
    emb = lm.model.get_input_embeddings().weight
    seg = Segmenter(lm.tokenizer)
    out = dict(args=vars(args), env=env_info(lm), by_lang={})
    last = lm.n_layers
    for lang in args.langs:
        fit = load_records(args.model, lang, "fit")
        ev = load_records(args.model, lang, "eval")
        vocab = ahead_vocab(fit, size=args.vocab)
        print(f"\n[{lang}] vocab {len(vocab)}; building EAGLE sequence data...")
        ST, TOK, Y, TM, FM = eagle_seq_data(fit, last, vocab, K=6)
        print(f"  training words: {len(ST)}")
        t0 = time.time()
        head = fit_word_eagle_rollout(ST, TOK, Y, TM, FM, emb, len(vocab), hidden=args.hidden, epochs=args.epochs)
        fit_s = time.time() - t0
        # current best baseline: word head (current state) + ahead, over the same vocab
        lex = Lexicon.from_records(fit, size=5000, min_count=2)
        big = Lexicon.from_texts(lm.tokenizer, load_texts(lang, "fit", 4000), size=30000, min_count=2)
        Xc, yc = head_training_data(fit, lex, last, mode="current")
        cur_head = fit_head(Xc, yc, len(lex), epochs=20)
        av = ahead_vocab_heads(fit, size=args.vocab)
        Xa, Ta, Ya = ahead_training_data(fit, last, av, K=4)
        ahead = fit_ahead(Xa, Ta, Ya, emb, len(av), epochs=12)
        hybrid = AheadDrafter(ahead, emb, av, primary=HeadDrafter(lex, cur_head, mode="current", fallback=DictionaryDrafter(big)))

        eagle = WordEagleDrafter(head, emb, vocab, max_draft=args.max_draft)
        eagle_fb = WordEagleDrafter(head, emb, vocab, max_draft=args.max_draft,
                                    primary=HeadDrafter(lex, cur_head, mode="current", fallback=DictionaryDrafter(big)))
        drafters = {"oracle": OracleDrafter(), "word_head_then_ahead": hybrid, "eagle": eagle, "eagle_then_head": eagle_fb}
        em = {}
        ev_emu = ev[:80]  # emulation signal; headline numbers come from real decoding
        for name, d in drafters.items():
            layer = None if name == "oracle" else last
            e = emulated_steps(ev_emu, d, layer=layer)
            em[name] = e
            print(f"  {name:>22}: {e['steps_per_token']:.3f} passes/token (x{e['step_reduction']:.2f}), "
                  f"accept {e['acceptance']:.1%}, whole-word {e['whole_word_first_draft']:.1%}")

        # real decoding
        texts = load_texts(lang, "eval", 300)[-args.prompts:]
        dec = {k: [] for k in ["plain", "word_head_then_ahead", "eagle", "eagle_then_head"]}
        lossless = True
        for i, t in enumerate(texts):
            ids = lm.tokenizer(t, add_special_tokens=False)["input_ids"][: args.prompt_tokens]
            ref = None
            for name in dec:
                d = None if name == "plain" else drafters[name]
                r = generate(lm, ids, args.new_tokens, drafter=d, layer=last, segmenter=seg, stop_ids=set())
                if ref is None:
                    ref = r.tokens
                elif r.tokens != ref:
                    lossless = False  # byte-identical to plain; divergences are fp ties (see diag_lossless.py)
                dec[name].append(dict(tokens=len(r.tokens), forwards=r.n_forward, seconds=r.seconds,
                                      drafted=int(sum(r.draft_lengths)), accepted=int(sum(r.accepted))))
            print(f"\r  decoding {i+1}/{len(texts)}", end="", flush=True)
        print()
        summ = {}
        base = sum(r["seconds"] for r in dec["plain"])
        for name, rs in dec.items():
            tok = sum(r["tokens"] for r in rs); fw = sum(r["forwards"] - 1 for r in rs)
            sec = sum(r["seconds"] for r in rs)
            summ[name] = dict(tokens_per_forward=tok / fw, wallclock_speedup=base / sec,
                              acceptance=sum(r["accepted"] for r in rs) / max(sum(r["drafted"] for r in rs), 1))
            print(f"  {name:>22}: {summ[name]['tokens_per_forward']:.2f} tok/fwd, "
                  f"accept {summ[name]['acceptance']:.1%}, wall x{summ[name]['wallclock_speedup']:.2f}")
        print(f"  lossless: {lossless}")
        torch.save(head.state_dict(), cache_path(args.model, lang, "eagle_head").with_suffix(".pt"))
        out["by_lang"][lang] = dict(emulation=em, decoding=summ, lossless=lossless, fit_seconds=fit_s,
                                    n_train=int(len(Y)), vocab=len(vocab))
    write_json(out, results_dir(args.model) / "eagle.json")


if __name__ == "__main__":
    main()
