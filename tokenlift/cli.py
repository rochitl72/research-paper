"""TokenLift command line: a headroom report, or a quick accelerated generation.

    tokenlift report --model Qwen/Qwen3-0.6B-Base --lang ta
    tokenlift gen    --model Qwen/Qwen3-0.6B-Base --lang ta --head head_ta.pt --prompt "..."
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser(prog="tokenlift")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("report", help="estimate acceleration headroom for a model on a text sample")
    r.add_argument("--model", required=True)
    r.add_argument("--text-file", required=True, help="UTF-8 file, one text sample per line")
    r.add_argument("--n", type=int, default=120)

    g = sub.add_parser("gen", help="accelerated generation from a trained head")
    g.add_argument("--model", required=True)
    g.add_argument("--lang", required=True)
    g.add_argument("--head", required=True)
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-new", type=int, default=64)
    g.add_argument("--temperature", type=float, default=0.0)

    a = ap.parse_args()
    from tokenlift import TokenLift
    spec = TokenLift.from_pretrained(a.model)

    if a.cmd == "report":
        from tokenlift.diagnostic import headroom
        with open(a.text_file, encoding="utf-8") as f:
            texts = [ln.strip() for ln in f if ln.strip()]
        print(json.dumps(headroom(spec.lm, texts, n=a.n), indent=2))
    elif a.cmd == "gen":
        spec.load_head(a.lang, a.head)
        if a.temperature > 0:
            out = spec.sample(a.prompt, max_new=a.max_new, temperature=a.temperature, lang=a.lang)
        else:
            out = spec.generate(a.prompt, max_new=a.max_new, lang=a.lang)
        print(out["text"])
        print("\n[tokens/forward: {:.2f}]".format(out.get("tokens_per_forward", float("nan"))))


if __name__ == "__main__":
    main()
