"""Kill test 3: word heads as drafters.

(a) Teacher-forced emulation on held-out natural text: how many sequential
    decoding steps would each drafter save? Dictionary (context-blind) vs word
    heads reading different layers, plus an upper bound.
(b) Real lossless decoding from held-out prompts: tokens per forward pass,
    wall-clock time, and an exact-match check against plain greedy decoding.
"""
import argparse
import time

import torch

from common import cache_path, env_info, load_records, load_texts, results_dir, start_log, write_json

from wordhead.heads import (DictionaryDrafter, HeadDrafter, Lexicon, OracleDrafter, emulated_steps, fit_head,
                            head_training_data)
from wordhead.models import load
from wordhead.specdec import generate
from wordhead.words import Segmenter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--langs", nargs="+", default=["ta", "hi", "en"])
    ap.add_argument("--lexicon-size", type=int, default=5000)
    ap.add_argument("--dictionary-size", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--prompts", type=int, default=12)
    ap.add_argument("--prompt-tokens", type=int, default=32)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args()
    start_log(f"04_heads_and_decoding_{args.model.split('/')[-1]}")
    out = dict(args=vars(args))
    lm = load(args.model, device=args.device, dtype=args.dtype)
    out["env"] = env_info(lm)
    for lang in args.langs:
        fit = load_records(args.model, lang, "fit")
        ev = load_records(args.model, lang, "eval")
        tok = lm.tokenizer
        lex = Lexicon.from_records(fit, size=args.lexicon_size, min_count=2)  # words the head is trained on
        big = Lexicon.from_texts(tok, load_texts(lang, "fit", 4000), size=args.dictionary_size, min_count=2)
        lex.save(cache_path(args.model, lang, "lexicon").with_suffix(".json"))
        layers = sorted(fit[0].preword_hidden)
        last = max(layers)
        res = dict(head_lexicon=len(lex), dictionary_lexicon=len(big), emulation={})

        def show(name, em):
            print(f"  {name:>22}: {em['steps_per_token']:.3f} forward passes per word token "
                  f"(x{em['step_reduction']:.2f} fewer), draft acceptance {em['acceptance']:.1%}, "
                  f"whole word on first draft {em['whole_word_first_draft']:.1%}")

        def coverage(lx):
            cov = sum(1 for r in ev for u in r.units if u.kind == "word" and u.n_tokens > 1 and u.ids in lx.index)
            tot = sum(1 for r in ev for u in r.units if u.kind == "word" and u.n_tokens > 1)
            return cov / max(tot, 1)

        res["coverage_head_lexicon"], res["coverage_dictionary"] = coverage(lex), coverage(big)
        print(f"\n[{lang}] head lexicon {len(lex)} words (covers {res['coverage_head_lexicon']:.1%} of held-out multi-token "
              f"words); dictionary {len(big)} words (covers {res['coverage_dictionary']:.1%})")
        em = res["emulation"]
        em["oracle"] = emulated_steps(ev, OracleDrafter(), layer=None)
        show("oracle (upper bound)", em["oracle"])
        dict_small, dict_big = DictionaryDrafter(lex), DictionaryDrafter(big)
        em["dictionary_small"] = emulated_steps(ev, dict_small, layer=None)
        show("dictionary (head lex)", em["dictionary_small"])
        em["dictionary"] = emulated_steps(ev, dict_big, layer=None)
        show("dictionary", em["dictionary"])
        heads = {}
        for L in layers:
            if L == 0:
                continue
            X, y = head_training_data(fit, lex, L, mode="preword")
            t0 = time.time()
            head = fit_head(X, y, len(lex), epochs=args.epochs)
            e = emulated_steps(ev, HeadDrafter(lex, head, mode="preword", fallback=dict_big), layer=L)
            e["fit_seconds"], e["n_train"] = time.time() - t0, int(len(y))
            em[f"preword_head_L{L}"] = e
            heads[L] = head
            show(f"pre-word head L{L}", e)
        X, y = head_training_data(fit, lex, last, mode="current")
        t0 = time.time()
        cur_head = fit_head(X, y, len(lex), epochs=args.epochs)
        e = emulated_steps(ev, HeadDrafter(lex, cur_head, mode="current", fallback=dict_big), layer=last)
        e["fit_seconds"], e["n_train"] = time.time() - t0, int(len(y))
        em[f"current_head_L{last}"] = e
        show(f"current-state head L{last}", e)
        best_L = min(heads, key=lambda L: em[f"preword_head_L{L}"]["steps"])
        res["best_layer"] = best_L
        torch.save(cur_head.state_dict(), cache_path(args.model, lang, f"current_head_L{last}").with_suffix(".pt"))

        # (b) real decoding
        seg = Segmenter(lm.tokenizer)
        texts = load_texts(lang, "eval", 300)[-args.prompts:]
        # real decoding reads the final layer (no extra hooks needed)
        drafters = {"plain": None, "dictionary": dict_big,
                    "preword_head": HeadDrafter(lex, heads[last], mode="preword", fallback=dict_big),
                    "current_head": HeadDrafter(lex, cur_head, mode="current", fallback=dict_big)}
        dec = {k: [] for k in drafters}
        lossless = True
        for i, t in enumerate(texts):
            ids = lm.tokenizer(t, add_special_tokens=False)["input_ids"][: args.prompt_tokens]
            ref = None
            for name, d in drafters.items():
                r = generate(lm, ids, args.new_tokens, drafter=d, layer=last, segmenter=seg, stop_ids=set())
                if ref is None:
                    ref = r.tokens
                elif r.tokens != ref:
                    lossless = False
                dec[name].append(dict(tokens=len(r.tokens), forwards=r.n_forward, seconds=r.seconds,
                                      drafts=len(r.draft_lengths), drafted=int(sum(r.draft_lengths)), accepted=int(sum(r.accepted))))
            print(f"\r  decoding prompt {i+1}/{len(texts)}", end="", flush=True)
        print()
        summary = {}
        for name, rs in dec.items():
            tok = sum(r["tokens"] for r in rs)
            fw = sum(r["forwards"] - 1 for r in rs)  # exclude the shared prompt pass
            sec = sum(r["seconds"] for r in rs)
            summary[name] = dict(tokens_per_forward=tok / fw, seconds=sec,
                                 acceptance=sum(r["accepted"] for r in rs) / max(sum(r["drafted"] for r in rs), 1))
        base = summary["plain"]["seconds"]
        for name in summary:
            summary[name]["wallclock_speedup"] = base / summary[name]["seconds"]
            print(f"  {name:>14}: {summary[name]['tokens_per_forward']:.2f} tokens/forward, "
                  f"acceptance {summary[name]['acceptance']:.1%}, wall-clock x{summary[name]['wallclock_speedup']:.2f}")
        print(f"  lossless (identical to plain greedy on all prompts): {lossless}")
        sample = lm.tokenizer.decode(generate(lm, lm.tokenizer(texts[0], add_special_tokens=False)["input_ids"][: args.prompt_tokens],
                                              args.new_tokens, drafter=None, segmenter=seg, stop_ids=set()).tokens)
        res["decoding"] = dict(summary=summary, per_prompt=dec, lossless=lossless, sample_generation=sample)
        out[lang] = res
    write_json(out, results_dir(args.model) / "heads_and_decoding.json")


if __name__ == "__main__":
    main()
