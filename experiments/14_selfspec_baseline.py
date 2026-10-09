"""G3.8 External baseline: standard (unrestricted) self-speculative decoding.

Draft model Qwen3-0.6B-Base proposes K tokens greedily; target Qwen3-4B-Base
verifies them in one forward pass and accepts the longest prefix equal to its
own greedy argmax (textbook greedy speculative decoding, Leviathan/Chen). This
is the obvious same-tokenizer baseline the word-scoped TokenLift head should be
compared against: it is NOT word-scoped, so it can draft across word boundaries.

We report tokens per target forward pass (the hardware-independent quality
metric the paper leads with), on the SAME prompts as the 4B EAGLE run, and
verify the output equals plain 4B greedy (losslessness sanity).

    python experiments/14_selfspec_baseline.py --target <path-to-4B> --draft Qwen/Qwen3-0.6B-Base
"""
import argparse, json, time, os
import torch
from common import load_texts, results_dir, env_info
from wordhead.models import load


def argmax_last(lm, seq, device):
    x = torch.tensor([seq], device=device)
    with torch.no_grad():
        lg = lm.model(x).logits[0, -1]
    return int(lg.argmax())


def spec_decode(draft, tgt, ids, n_new, K, tdev, ddev):
    seq = list(ids)
    tgt_passes = 0
    out = []
    while len(out) < n_new:
        d = []
        for _ in range(K):
            d.append(argmax_last(draft, seq + d, ddev))
        x = torch.tensor([seq + d], device=tdev)
        with torch.no_grad():
            tlog = tgt.model(x).logits[0]
        tgt_passes += 1
        accepted = []
        for j in range(K):
            pos = len(seq) + j - 1
            ttok = int(tlog[pos].argmax())
            if ttok == d[j]:
                accepted.append(d[j])
            else:
                accepted.append(ttok)
                break
        else:
            accepted.append(int(tlog[len(seq) + K - 1].argmax()))
        seq += accepted
        out += accepted
    return out[:n_new], tgt_passes


def plain_greedy(tgt, ids, n_new, tdev):
    seq = list(ids)
    for _ in range(n_new):
        seq.append(argmax_last(tgt, seq, tdev))
    return seq[len(ids):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--langs", nargs="+", default=["ta", "hi"])
    ap.add_argument("--prompts", type=int, default=24)
    ap.add_argument("--prompt-tokens", type=int, default=32)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--K", type=int, default=5)
    args = ap.parse_args()

    tgt = load(args.target, device="auto", dtype="float16")
    draft = load(args.draft, device="cpu", dtype="float32")
    tdev = next(tgt.model.parameters()).device
    ddev = next(draft.model.parameters()).device
    print(f"target on {tdev}, draft on {ddev}, K={args.K}", flush=True)

    out = {"args": vars(args), "env": env_info(tgt), "by_lang": {}}
    for lang in args.langs:
        texts = load_texts(lang, "eval", 300)[-args.prompts:]
        tgt_fwd_total = 0
        new_total = 0
        exact = 0
        t0 = time.time()
        for i, t in enumerate(texts):
            ids = tgt.tokenizer(t, add_special_tokens=False)["input_ids"][: args.prompt_tokens]
            toks, passes = spec_decode(draft, tgt, ids, args.new_tokens, args.K, tdev, ddev)
            ref = plain_greedy(tgt, ids, args.new_tokens, tdev)
            tgt_fwd_total += passes
            new_total += len(toks)
            exact += int(toks == ref)
            print(f"  [{lang}] {i+1}/{len(texts)}  tok/fwd {new_total/max(tgt_fwd_total,1):.2f}  exact {exact}/{i+1}", flush=True)
        tpf = new_total / max(tgt_fwd_total, 1)
        out["by_lang"][lang] = dict(tokens_per_target_forward=tpf, exact_match_vs_plain=exact,
                                    n_prompts=len(texts), K=args.K, seconds=time.time()-t0)
        print(f"[{lang}] DONE self-spec tok/target-fwd = {tpf:.2f} ; exact {exact}/{len(texts)}", flush=True)
    rd = results_dir(args.target)
    os.makedirs(rd, exist_ok=True)
    with open(os.path.join(str(rd), "selfspec_baseline.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote selfspec_baseline.json", flush=True)


if __name__ == "__main__":
    main()
