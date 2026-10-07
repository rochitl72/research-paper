"""Is the non-losslessness a bug or floating-point near-ties?"""
import torch
from common import load_records, load_texts
from wordhead.models import load
from wordhead.specdec import generate
from wordhead.words import Segmenter
from wordhead.eagle import WordEagleDrafter, ahead_vocab, eagle_seq_data, fit_word_eagle_rollout

lm = load("Qwen/Qwen3-0.6B-Base")
emb = lm.model.get_input_embeddings().weight
seg = Segmenter(lm.tokenizer)
last = lm.n_layers
fit = load_records("Qwen/Qwen3-0.6B-Base", "ta", "fit")
vocab = ahead_vocab(fit, size=4000)
ST, TOK, Y, TM, FM = eagle_seq_data(fit, last, vocab, K=6)
head = fit_word_eagle_rollout(ST, TOK, Y, TM, FM, emb, len(vocab), epochs=20)
eagle = WordEagleDrafter(head, emb, vocab, max_draft=8)

texts = load_texts("ta", "eval", 300)[-24:]
diverge = 0
@torch.no_grad()
def plain_logits_gap(ids):
    out = lm.model(input_ids=torch.tensor([ids], device=lm.device))
    lg = out.logits[0, -1]
    top2 = torch.topk(lg, 2).values
    return float(top2[0] - top2[1])

for i, t in enumerate(texts):
    ids = lm.tokenizer(t, add_special_tokens=False)["input_ids"][:32]
    pl = generate(lm, ids, 64, drafter=None, layer=last, segmenter=seg, stop_ids=set()).tokens
    eg = generate(lm, ids, 64, drafter=eagle, layer=last, segmenter=seg, stop_ids=set()).tokens
    if pl != eg:
        j = next(k for k in range(min(len(pl), len(eg))) if pl[k] != eg[k])
        gap = plain_logits_gap(ids + pl[:j])
        print(f"prompt {i}: diverge at token {j}/{len(pl)}; plain={pl[j]} eagle={eg[j]}; top1-top2 logit gap = {gap:.4f}")
        diverge += 1
print(f"\n{diverge}/{len(texts)} prompts diverged")
