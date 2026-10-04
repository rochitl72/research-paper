"""Command-line interface.

    wordhead audit    --model M --text-file ta.txt            # spelling share report
    wordhead fit      --model M --text-file ta.txt --out dir  # lexicon + word head
    wordhead generate --model M --head-dir dir --prompt "..."  # lossless faster decoding
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _read_texts(path: str, limit: int) -> list[str]:
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                line = json.loads(line).get("text", "")
            lines.append(line)
            if len(lines) >= limit:
                break
    return lines


def cmd_audit(a):
    from .corpus import run_corpus
    from .models import load
    from .spelling import spelling_summary, word_table

    lm = load(a.model, device=a.device, dtype=a.dtype)
    recs = run_corpus(lm, _read_texts(a.text_file, a.limit), max_len=a.max_len, batch_size=a.batch_size)
    s = spelling_summary(word_table(recs))
    print(json.dumps({k: v for k, v in s.items()}, indent=2, ensure_ascii=False))
    if a.out:
        Path(a.out).write_text(json.dumps(s, indent=2, ensure_ascii=False))


def cmd_fit(a):
    from .corpus import run_corpus
    from .heads import Lexicon, fit_head, head_training_data
    from .models import load

    lm = load(a.model, device=a.device, dtype=a.dtype)
    layer = a.layer if a.layer >= 0 else lm.n_layers + 1 + a.layer
    recs = run_corpus(lm, _read_texts(a.text_file, a.limit), max_len=a.max_len, batch_size=a.batch_size,
                      hidden_layers=[layer])
    lex = Lexicon.from_records(recs, size=a.lexicon_size, min_count=2)
    X, y = head_training_data(recs, lex, layer)
    head = fit_head(X, y, len(lex), epochs=a.epochs, verbose=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lex.save(out / "lexicon.json")
    torch.save(head.state_dict(), out / "head.pt")
    (out / "config.json").write_text(json.dumps(dict(model=a.model, layer=layer, d_model=lm.d_model, n_words=len(lex))))
    print(f"saved lexicon ({len(lex)} words) and head (layer {layer}) to {out}")


def cmd_generate(a):
    from .heads import DictionaryDrafter, HeadDrafter, Lexicon, WordHead
    from .models import load
    from .specdec import generate

    lm = load(a.model, device=a.device, dtype=a.dtype)
    ids = lm.tokenizer(a.prompt, add_special_tokens=False)["input_ids"]
    drafter, layer = None, -1
    if a.head_dir:
        d = Path(a.head_dir)
        cfg = json.loads((d / "config.json").read_text())
        lex = Lexicon.load(d / "lexicon.json")
        if a.drafter == "dictionary":
            drafter = DictionaryDrafter(lex)
        else:
            head = WordHead(cfg["d_model"], cfg["n_words"])
            head.load_state_dict(torch.load(d / "head.pt"))
            drafter, layer = HeadDrafter(lex, head.eval()), cfg["layer"]
    plain = generate(lm, ids, a.max_new, drafter=None)
    fast = generate(lm, ids, a.max_new, drafter=drafter, layer=layer)
    print(lm.tokenizer.decode(fast.tokens))
    print(f"\nidentical to greedy: {fast.tokens == plain.tokens} | tokens/forward: plain {plain.tokens_per_forward:.2f}, "
          f"{a.drafter} {fast.tokens_per_forward:.2f} | seconds: {plain.seconds:.2f} -> {fast.seconds:.2f}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="wordhead")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--model", required=True)
        sp.add_argument("--device", default="auto")
        sp.add_argument("--dtype", default="auto")

    s = sub.add_parser("audit", help="spelling share of a model on a text file")
    common(s)
    s.add_argument("--text-file", required=True)
    s.add_argument("--limit", type=int, default=500)
    s.add_argument("--max-len", type=int, default=256)
    s.add_argument("--batch-size", type=int, default=4)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_audit)

    s = sub.add_parser("fit", help="build a lexicon and fit a word head")
    common(s)
    s.add_argument("--text-file", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--layer", type=int, default=-1)
    s.add_argument("--limit", type=int, default=2000)
    s.add_argument("--max-len", type=int, default=256)
    s.add_argument("--batch-size", type=int, default=4)
    s.add_argument("--lexicon-size", type=int, default=5000)
    s.add_argument("--epochs", type=int, default=20)
    s.set_defaults(fn=cmd_fit)

    s = sub.add_parser("generate", help="lossless word-drafted greedy generation")
    common(s)
    s.add_argument("--prompt", required=True)
    s.add_argument("--head-dir")
    s.add_argument("--drafter", choices=["head", "dictionary"], default="head")
    s.add_argument("--max-new", type=int, default=64)
    s.set_defaults(fn=cmd_generate)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
