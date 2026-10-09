"""G3.9 Logit-lens baseline for the plan probe.

Reviewer concern: the probe's baseline is a context-blind dictionary, so "the
hidden state beats it" might only show that context helps. The right baseline is
the MODEL'S OWN distribution: does the pre-word hidden state encode the word
beyond what the model itself predicts?

We add exactly that. Within each first-token group, we rank the candidate words
by the model's own teacher-forced likelihood of the whole word given the context
up to the pre-word position (the strongest single-position logit-lens readout),
and report its top-1 accuracy next to the trained probe and the majority
(dictionary) baseline. Because group members share the first token, the model's
one-step next-token output cannot separate them; this quantifies how much the
probe recovers that the model has not yet surfaced in its own predictions.
"""
import argparse, json, random
from collections import Counter, defaultdict
import numpy as np
import torch
from common import load_records, results_dir
from wordhead.models import load


def groups_and_examples(records, min_count=4, min_per_group=2, max_classes=1000):
    occ = []  # (si, start, word_ids)
    for si, rec in enumerate(records):
        for ui in rec.preword_units:
            u = rec.units[ui]
            if u.n_tokens < 2:
                continue
            occ.append((si, u.start, tuple(u.ids)))
    cnt = Counter(w for _, _, w in occ)
    bygroup = defaultdict(set)
    for w, c in cnt.items():
        if c >= min_count:
            bygroup[w[0]].add(w)
    keep = {w for g in bygroup.values() if len(g) >= min_per_group for w in g}
    keep = set(sorted(keep, key=lambda w: -cnt[w])[:max_classes])
    group_members = defaultdict(list)
    for w in keep:
        group_members[w[0]].append(w)
    majority = {ft: max(ms, key=lambda w: cnt[w]) for ft, ms in group_members.items()}
    ex = [(si, st, w) for si, st, w in occ if w in keep]
    return ex, group_members, majority, cnt


@torch.no_grad()
def word_logprob(lm, ctx_ids, word_ids, dev):
    """Teacher-forced sum log p(word | ctx); softmax only at the needed positions."""
    seq = list(ctx_ids) + list(word_ids)
    x = torch.tensor([seq], device=dev)
    logits = lm.model(x).logits[0]
    pos = [len(ctx_ids) + k - 1 for k in range(len(word_ids))]
    rows = logits[pos].float()                      # [len(word), V]
    lp = torch.log_softmax(rows, -1)
    toks = torch.tensor(word_ids, device=lp.device)
    return float(lp.gather(1, toks[:, None]).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--langs", nargs="+", default=["ta", "hi"])
    ap.add_argument("--max-examples", type=int, default=400)
    ap.add_argument("--max-candidates", type=int, default=40)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    lm = load(args.model, device=args.device, dtype="float32")
    dev = next(lm.model.parameters()).device
    out = {"args": vars(args), "by_lang": {}}
    rng = random.Random(0)
    for lang in args.langs:
        recs = load_records(args.model, lang, "fit") + load_records(args.model, lang, "eval")
        ex, group_members, majority, cnt = groups_and_examples(recs)
        rng.shuffle(ex)
        ex = ex[: args.max_examples]
        ll_correct = maj_correct = n = 0
        for si, st, w in ex:
            ctx = recs[si].ids[:st]
            cands = group_members[w[0]]
            if len(cands) > args.max_candidates:
                cands = sorted(cands, key=lambda c: -cnt[c])[: args.max_candidates]
                if w not in cands:
                    cands = cands[:-1] + [w]
            scores = {c: word_logprob(lm, ctx, c, dev) for c in cands}
            pred = max(scores, key=scores.get)
            ll_correct += int(pred == w)
            maj_correct += int(majority[w[0]] == w)
            n += 1
        out["by_lang"][lang] = dict(n=n, logit_lens_top1=ll_correct / n,
                                    majority_top1=maj_correct / n,
                                    mean_candidates=np.mean([len(group_members[w[0]]) for _, _, w in ex]))
        print(f"[{lang}] n={n}  logit-lens top1={ll_correct/n:.3f}  majority={maj_correct/n:.3f}", flush=True)
    with open(str(results_dir(args.model)) + "/logitlens_probe.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote logitlens_probe.json", flush=True)


if __name__ == "__main__":
    main()
