"""G2.7 Validate the headroom diagnostic.

The diagnostic predicts best tokens/forward as tpw / (1 + (1-g)*(tpw-1)),
from tokens-per-word (tpw) and the model's greedy-correct continuation
fraction (g). We check that prediction against the oracle upper bound actually
measured by the drafter emulation (emulation.oracle.step_reduction), across
three models x two languages, including the low-fragmentation XGLM cases that
should (and do) predict little headroom. Reads only results/ JSON.
"""
import json, os, math

MODELS = ["Qwen3-0.6B-Base", "Qwen3-1.7B-Base", "xglm-564M"]
LANGS = ["ta", "hi"]
ROOT = os.path.join(os.path.dirname(__file__), "..", "results")


def est_tpf(tpw, g):
    return tpw / max(1 + (1 - g) * max(tpw - 1, 0), 1e-6)


def pearson(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    vx = sum((x-mx)**2 for x in xs); vy = sum((y-my)**2 for y in ys)
    return cov / math.sqrt(vx*vy)


rows = []
for m in MODELS:
    sp = json.load(open(os.path.join(ROOT, m, "spelling_share.json")))
    hd = json.load(open(os.path.join(ROOT, m, "heads_and_decoding.json")))
    for lg in LANGS:
        tpw = sp[lg]["tokens_per_word"]; g = sp[lg]["cont_greedy_correct"]
        pred = est_tpf(tpw, g)
        oracle = hd[lg]["emulation"]["oracle"]["step_reduction"]
        rows.append((m.replace("Qwen3-", "").replace("-Base", ""), lg, tpw, g, pred, oracle))

print(f"{'model':8} {'lg':3} {'tpw':>6} {'g':>6} {'pred':>6} {'oracle':>7}")
for r in rows:
    print(f"{r[0]:8} {r[1]:3} {r[2]:6.2f} {r[3]:6.3f} {r[4]:6.2f} {r[5]:7.2f}")

pred = [r[4] for r in rows]; oracle = [r[5] for r in rows]
mae = sum(abs(p-o) for p, o in zip(pred, oracle))/len(rows)
print(f"\nn = {len(rows)} (model x lang points, incl. low-fragmentation XGLM)")
print(f"Pearson r(pred, oracle tok/fwd) = {pearson(pred, oracle):.3f}")
print(f"MAE = {mae:.2f} tok/fwd ; mean oracle = {sum(oracle)/len(oracle):.2f}")
