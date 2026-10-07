"""Statistical significance for the mechanism and decoding results.

Reads the raw per-item data already stored by 03_patching, 03b_knockout,
04_heads_and_decoding and 02_plan_probe, and computes bootstrap 95% CIs and
paired nonparametric tests. No model is loaded here. Writes
results/<model>/significance.json and prints a readable report.

    python experiments/07_significance.py --models Qwen/Qwen3-0.6B-Base Qwen/Qwen3-1.7B-Base
"""
import argparse
import json

import numpy as np
from scipy import stats

from common import RESULTS, slug

LANG = {"ta": "Tamil", "hi": "Hindi", "en": "English"}
RNG = np.random.default_rng(0)
B = 10000


def load(model, name):
    p = RESULTS / slug(model) / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def boot_ci(x, stat=np.mean, n=B, alpha=0.05):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"))
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    bs = stat(x[idx], axis=1)
    lo, hi = np.percentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(x)), float(lo), float(hi)


def paired_boot_ci(a, b, stat=np.mean, n=B, alpha=0.05):
    """CI on stat(a) - stat(b) over paired items (resample pairs together)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = RNG.integers(0, len(a), size=(n, len(a)))
    d = stat(a[idx], axis=1) - stat(b[idx], axis=1)
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(a) - stat(b)), float(lo), float(hi)


def mcnemar(a_success, b_success):
    """Exact McNemar on paired binary outcomes. Returns (b, c, p_two_sided)."""
    a = np.asarray(a_success, bool)
    b = np.asarray(b_success, bool)
    n_b = int(np.sum(a & ~b))  # a right, b wrong
    n_c = int(np.sum(~a & b))  # a wrong, b right
    p = stats.binomtest(min(n_b, n_c), n_b + n_c, 0.5).pvalue if (n_b + n_c) else 1.0
    return n_b, n_c, float(p)


def fmt(ci):
    return f"{ci[0]:.3f} [{ci[1]:.3f}, {ci[2]:.3f}]"


def do_patching(pa):
    out = {}
    for lang in ("ta", "hi"):
        if lang not in pa:
            continue
        pre, ft = pa[lang]["preword"], pa[lang]["first_token"]
        layers = pre["layers"]
        eff_pre = np.array([r["effect"] for r in pre["per_pair"]])   # [pairs, layers]
        tx_pre = np.array([r["transfer_x"] for r in pre["per_pair"]], float)
        tx_ft = np.array([r["transfer_x"] for r in ft["per_pair"]], float)
        li_best = int(np.nanargmax(np.nanmedian(eff_pre, 0)))
        li_last = len(layers) - 1
        # bootstrap CI on the pre-word median effect at its best layer
        eff_best = eff_pre[:, li_best]
        eff_best = eff_best[~np.isnan(eff_best)]
        ci_eff = boot_ci(eff_best, np.median)
        # paired transfer at the last layer: first-token vs pre-word
        nb, nc, p = mcnemar(tx_ft[:, li_last] > 0.5, tx_pre[:, li_last] > 0.5)
        ci_tx = paired_boot_ci(tx_ft[:, li_last], tx_pre[:, li_last], np.mean)
        out[lang] = dict(
            n_pairs=len(eff_pre), best_layer=layers[li_best], last_layer=layers[li_last],
            preword_median_effect_ci=ci_eff,
            transfer_first_token_last=float(tx_ft[:, li_last].mean()),
            transfer_preword_last=float(tx_pre[:, li_last].mean()),
            transfer_diff_ci=ci_tx, mcnemar_b=nb, mcnemar_c=nc, mcnemar_p=p,
        )
    return out


def do_knockout(ko):
    out = {}
    for lang in ("ta", "hi"):
        if lang not in ko:
            continue
        per = ko[lang]["per_word"]
        bits = {k: np.array([r[k] for r in per]) / np.log(2) for k in ("preword", "preword_minus_1", "random_context")}
        oks = {k: np.array([r[k + "_ok"] for r in per], bool) for k in ("preword", "preword_minus_1", "random_context")}
        w = stats.wilcoxon(bits["preword"], bits["random_context"])
        nb, nc, p = mcnemar(oks["random_context"], oks["preword"])  # does blocking pre-word break more?
        out[lang] = dict(
            n_words=len(per),
            preword_bits_ci=boot_ci(bits["preword"], np.mean),
            random_bits_ci=boot_ci(bits["random_context"], np.mean),
            preword_vs_random_bits_ci=paired_boot_ci(bits["preword"], bits["random_context"], np.mean),
            wilcoxon_p=float(w.pvalue),
            still_spelled_preword=float(oks["preword"].mean()),
            still_spelled_random=float(oks["random_context"].mean()),
            mcnemar_b=nb, mcnemar_c=nc, mcnemar_p=p,
        )
    return out


def do_decoding(hd):
    out = {}
    for lang in ("ta", "hi"):
        if lang not in hd:
            continue
        pp = hd[lang]["decoding"]["per_prompt"]
        plain_sec = np.array([r["seconds"] for r in pp["plain"]])
        res = {}
        for name, rows in pp.items():
            tok = np.array([r["tokens"] for r in rows], float)
            fw = np.array([r["forwards"] - 1 for r in rows], float)
            tpf = tok / fw
            sec = np.array([r["seconds"] for r in rows])
            speed = plain_sec / sec
            entry = dict(n_prompts=len(rows), tokens_per_forward_ci=boot_ci(tpf, np.mean),
                         wallclock_speedup_ci=boot_ci(speed, np.mean))
            if name != "plain":
                # sign test: is the drafter faster than plain on a majority of prompts?
                wins = int(np.sum(speed > 1.0))
                entry["faster_than_plain"] = f"{wins}/{len(speed)}"
                entry["sign_p"] = float(stats.binomtest(wins, len(speed), 0.5, alternative="greater").pvalue)
            res[name] = entry
        out[lang] = res
    return out


def do_probe(pr):
    out = {}
    for lang in ("ta", "hi"):
        d = (pr.get(lang) or {}).get("by_layer") if pr else None
        if not d:
            continue
        last = d[-1]
        # 3 seeds: 95% CI via t(2). gap over control in SD units.
        acc, sd = last["acc"], last.get("acc_sd", 0.0)
        tcrit = 4.303  # t_{0.975, df=2}
        half = tcrit * sd / np.sqrt(3) if sd else 0.0
        out[lang] = dict(layer=last["layer"], acc=acc, acc_sd=sd,
                         acc_ci=[acc - half, acc + half],
                         control=last["acc_control"], majority=last["acc_majority"],
                         gap_over_control=acc - last["acc_control"],
                         gap_in_sd=(acc - last["acc_control"]) / sd if sd else float("inf"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen3-0.6B-Base"])
    args = ap.parse_args()
    for m in args.models:
        print(f"\n{'='*70}\n{slug(m)}\n{'='*70}")
        pr, pa, ko, hd = (load(m, n) for n in ("plan_probe", "patching", "knockout", "heads_and_decoding"))
        res = {}
        if pr:
            res["probe"] = do_probe(pr)
            for l, d in res["probe"].items():
                print(f"[probe {LANG[l]}] layer {d['layer']}: acc {d['acc']:.3f} "
                      f"CI[{d['acc_ci'][0]:.3f},{d['acc_ci'][1]:.3f}] vs control {d['control']:.3f} "
                      f"(+{d['gap_over_control']:.3f}, {d['gap_in_sd']:.0f} SD)")
        if pa:
            res["patching"] = do_patching(pa)
            for l, d in res["patching"].items():
                print(f"[patching {LANG[l]}] n={d['n_pairs']} pairs; pre-word median effect @L{d['best_layer']} "
                      f"{fmt(d['preword_median_effect_ci'])}; last-layer transfer first-token {d['transfer_first_token_last']:.2f} "
                      f"vs pre-word {d['transfer_preword_last']:.2f}, diff {fmt(d['transfer_diff_ci'])}, "
                      f"McNemar b={d['mcnemar_b']} c={d['mcnemar_c']} p={d['mcnemar_p']:.2e}")
        if ko:
            res["knockout"] = do_knockout(ko)
            for l, d in res["knockout"].items():
                print(f"[knockout {LANG[l]}] n={d['n_words']}; pre-word {fmt(d['preword_bits_ci'])} bits vs random "
                      f"{fmt(d['random_bits_ci'])}; paired diff {fmt(d['preword_vs_random_bits_ci'])}, "
                      f"Wilcoxon p={d['wilcoxon_p']:.2e}; still-spelled {d['still_spelled_preword']:.2f} vs "
                      f"{d['still_spelled_random']:.2f}, McNemar p={d['mcnemar_p']:.2e}")
        if hd:
            res["decoding"] = do_decoding(hd)
            for l, d in res["decoding"].items():
                best = max((k for k in d if k != "plain"), key=lambda k: d[k]["tokens_per_forward_ci"][0])
                e = d[best]
                print(f"[decoding {LANG[l]}] best drafter '{best}': tokens/forward {fmt(e['tokens_per_forward_ci'])}, "
                      f"speedup {fmt(e['wallclock_speedup_ci'])}, faster on {e['faster_than_plain']} prompts "
                      f"(sign p={e['sign_p']:.2e})")
        (RESULTS / slug(m) / "significance.json").write_text(json.dumps(res, indent=2))
        print("wrote", RESULTS / slug(m) / "significance.json")


if __name__ == "__main__":
    main()
