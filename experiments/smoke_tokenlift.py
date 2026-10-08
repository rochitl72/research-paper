"""End-to-end smoke test for the TokenLift package: load a saved head, run
lossless greedy generate (accelerated vs plain), lossless speculative sample,
and the headroom diagnostic."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenlift import TokenLift
from tokenlift.diagnostic import headroom
from experiments.common import load_texts

MODEL = "Qwen/Qwen3-0.6B-Base"
HEAD = os.path.expanduser("~/wordhead-research/cache/Qwen3-0.6B-Base/ta_eagle_head.pt")

# real Tamil text from the held-out eval corpus
texts = load_texts("ta", "eval", 40)
prompt = texts[0][:120]
print("PROMPT:", prompt[:80], "...\n")

spec = TokenLift.from_pretrained(MODEL)
spec.load_head("ta", HEAD)
print("head loaded: vocab=%d\n" % len(spec.vocabs["ta"]))

# 1) greedy, accelerated vs plain
t = time.time(); acc = spec.generate(prompt, max_new=48, lang="ta", accelerate=True); ta_acc = time.time() - t
t = time.time(); pla = spec.generate(prompt, max_new=48, lang="ta", accelerate=False); ta_pla = time.time() - t
same = acc["tokens"] == pla["tokens"]
print("GREEDY accelerated: %.2f tok/fwd, %d forwards, %.2fs" % (acc["tokens_per_forward"], acc["n_forward"], ta_acc))
print("GREEDY plain:       %.2f tok/fwd, %d forwards, %.2fs" % (pla["tokens_per_forward"], pla["n_forward"], ta_pla))
print("identical to plain greedy:", same)
if not same:
    diff = [i for i,(a,b) in enumerate(zip(acc["tokens"], pla["tokens"])) if a != b]
    print("  first divergence at token index:", diff[:3], "(fp tie expected)")
print("TEXT:", acc["text"][:120].replace("\n", " "), "\n")

# 2) lossless speculative sampling
t = time.time(); smp = spec.sample(prompt, max_new=48, temperature=0.8, lang="ta", seed=0); ts = time.time() - t
print("SAMPLE T=0.8: %.2f tok/fwd, accept %.1f%%, %d forwards, %.2fs" % (
    smp["tokens_per_forward"], 100 * smp["acceptance"], smp["n_forward"], ts))
print("TEXT:", smp["text"][:120].replace("\n", " "), "\n")

# 3) diagnostic
h = headroom(spec.lm, texts, n=40)
print("HEADROOM (ta):", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in h.items()})
print("\nSMOKE OK")
