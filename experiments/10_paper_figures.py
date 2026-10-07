"""Publication figures for the paper: charts A1-A8 and diagrams B1-B5.
Outputs vector PDFs into figures/paper/. Reads results/<model>/*.json only.

    python experiments/10_paper_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from common import RESULTS, ROOT, slug

OUT = ROOT / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

# validated categorical palette (reused from 05_figures.py)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#1c2733", "#5b6b7b", "#d7dee6", "#f4f7fa"
MODELS = ["Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base", "facebook/xglm-564M"]
MNAME = {"Qwen/Qwen3-0.6B-Base": "Qwen3-0.6B", "Qwen/Qwen3-1.7B-Base": "Qwen3-1.7B", "facebook/xglm-564M": "XGLM-564M"}
MCOL = {"Qwen/Qwen3-0.6B-Base": BLUE, "Qwen/Qwen3-1.7B-Base": ORANGE, "facebook/xglm-564M": AQUA}
LANGS = ["ta", "hi", "en"]
LNAME = {"ta": "Tamil", "hi": "Hindi", "en": "English"}
LCOL = {"ta": BLUE, "hi": ORANGE, "en": AQUA}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
    "axes.edgecolor": INK2, "axes.linewidth": 0.8, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK,
})


def load(model, name):
    p = RESULTS / slug(model) / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=150)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


# ---- A1: first-token information share, grouped bars (model x language) ----
def a1_first_token_share():
    sp = {m: load(m, "spelling_share") for m in MODELS}
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(LANGS))
    w = 0.26
    for i, m in enumerate(MODELS):
        vals = [100 * sp[m][l]["share_bits_in_first_token"] for l in LANGS]
        ax.bar(x + (i - 1) * w, vals, w, label=MNAME[m], color=MCOL[m], edgecolor="white", linewidth=0.6)
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([LNAME[l] for l in LANGS])
    ax.set_ylabel("First-token information share (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Where a word's information lives: share carried by its first token", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left")
    save(fig, "A1_first_token_share")


# ---- A2: fragmentation -> first-token share (scatter, the "law") ----
def a2_fragmentation_law():
    sp = {m: load(m, "spelling_share") for m in MODELS}
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    for m in MODELS:
        for l in LANGS:
            x = sp[m][l]["tokens_per_word"]; y = 100 * sp[m][l]["share_bits_in_first_token"]
            ax.scatter(x, y, s=90, color=LCOL[l], marker={"Qwen/Qwen3-0.6B-Base": "o", "Qwen/Qwen3-1.7B-Base": "s", "facebook/xglm-564M": "^"}[m],
                       edgecolor=INK, linewidth=0.6, zorder=3)
            ax.annotate(f"{MNAME[m].split('-')[0][:4]}·{LNAME[l][:2]}", (x, y), (4, 4), textcoords="offset points", fontsize=6.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("Tokens per word (log scale)  —  fragmentation")
    ax.set_ylabel("First-token information share (%)")
    ax.set_title("The more a tokenizer fragments a word,\nthe less its first token reveals", fontsize=10.5)
    # legend proxies
    from matplotlib.lines import Line2D
    langleg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=LCOL[l], markeredgecolor=INK, markersize=8, label=LNAME[l]) for l in LANGS]
    mkleg = [Line2D([0], [0], marker=mk, color="w", markerfacecolor=INK2, markeredgecolor=INK, markersize=8, label=nm)
             for mk, nm in [("o", "Qwen3-0.6B"), ("s", "Qwen3-1.7B"), ("^", "XGLM-564M")]]
    ax.legend(handles=langleg + mkleg, frameon=False, fontsize=7.5, ncol=2, loc="center right")
    ax.set_ylim(0, 105)
    save(fig, "A2_fragmentation_law")


# ---- A3: decoder speedup vs fragmentation ----
def a3_speedup_vs_fragmentation():
    sp = {m: load(m, "spelling_share") for m in MODELS}
    hd = {m: load(m, "heads_and_decoding") for m in MODELS}
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    for m in MODELS:
        for l in ("ta", "hi"):
            dec = hd[m][l]["decoding"]["summary"]
            best = max((k for k in dec if k != "plain"), key=lambda k: dec[k]["tokens_per_forward"])
            x = sp[m][l]["tokens_per_word"]; y = dec[best]["wallclock_speedup"]
            ax.scatter(x, y, s=90, color=LCOL[l], marker={"Qwen/Qwen3-0.6B-Base": "o", "Qwen/Qwen3-1.7B-Base": "s", "facebook/xglm-564M": "^"}[m],
                       edgecolor=INK, linewidth=0.6, zorder=3)
            ax.annotate(f"{MNAME[m].split('-')[-1]}·{LNAME[l][:2]}", (x, y), (4, 4), textcoords="offset points", fontsize=6.5, color=INK2)
    ax.axhline(1.0, color=INK2, ls="--", lw=0.9)
    ax.text(ax.get_xlim()[1], 1.0, " no speedup", va="center", ha="right", fontsize=7.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("Tokens per word (log scale)  —  fragmentation")
    ax.set_ylabel("Best wall-clock speedup (×)")
    ax.set_title("The lossless decoder helps most\nexactly where tokenization fragments the word", fontsize=10.5)
    from matplotlib.lines import Line2D
    langleg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=LCOL[l], markeredgecolor=INK, markersize=8, label=LNAME[l]) for l in ("ta", "hi")]
    ax.legend(handles=langleg, frameon=False, fontsize=8, loc="upper left")
    save(fig, "A3_speedup_vs_fragmentation")


# ---- A4: patching heatmap (layers x position) per model, Tamil ----
def a4_patching_heatmap():
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.6), sharey=True)
    for ax, m in zip(axes, MODELS):
        pa = load(m, "patching")
        if not pa or "ta" not in pa:
            ax.set_visible(False); continue
        layers = pa["ta"]["preword"]["layers"]
        mat = np.array([pa["ta"]["preword"]["effect_median"], pa["ta"]["first_token"]["effect_median"]]).T
        im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0, vmax=1, origin="lower")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pre-word", "first-token"], rotation=20, fontsize=8)
        ax.set_yticks(range(0, len(layers), 2)); ax.set_yticklabels([layers[i] for i in range(0, len(layers), 2)], fontsize=7)
        ax.set_title(MNAME[m], fontsize=9.5)
        ax.grid(False)
    axes[0].set_ylabel("Layer")
    fig.suptitle("Causal effect of patching each position toward the donor word (Tamil)", fontsize=10.5, y=1.02)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("normalized logit-difference effect", fontsize=8)
    save(fig, "A4_patching_heatmap")


# ---- A5: probe by layer with error bars (Tamil, Hindi) ----
def a5_probe_by_layer():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=False)
    for ax, l in zip(axes, ("ta", "hi")):
        for m in MODELS:
            pr = load(m, "plan_probe")
            by = pr[l]["by_layer"]
            xs = [r["layer"] for r in by]
            ys = [100 * r["acc"] for r in by]
            es = [100 * r.get("acc_sd", 0) for r in by]
            ax.errorbar(xs, ys, yerr=es, marker="o", ms=4, lw=1.6, capsize=2.5, color=MCOL[m], label=MNAME[m])
        # control band (use last model's control as representative dashed line)
        ctrl = 100 * by[-1]["acc_control"]; maj = 100 * by[-1]["acc_majority"]
        ax.axhline(maj, color=INK2, ls=":", lw=1.0)
        ax.axhline(ctrl, color="#b23b3b", ls="--", lw=1.0)
        ax.text(xs[-1], maj, " dict. prior", va="bottom", ha="right", fontsize=7, color=INK2)
        ax.text(xs[-1], ctrl, " shuffled control", va="top", ha="right", fontsize=7, color="#b23b3b")
        ax.set_title(LNAME[l], fontsize=10); ax.set_xlabel("Layer")
    axes[0].set_ylabel("Probe accuracy (%)  (1000 words)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("The upcoming word is linearly readable before it is written (±1 SD over 3 seeds)", fontsize=10.5, y=1.02)
    save(fig, "A5_probe_by_layer")


# ---- A6: knockout with seed error bars ----
def a6_knockout():
    ks = {m: load(m, "knockout_seeds") for m in MODELS}
    ko = {m: load(m, "knockout") for m in MODELS}
    conds = [("preword", "pre-word"), ("preword_minus_1", "neighbour"), ("random_context", "random")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    for ax, l in zip(axes, ("ta", "hi")):
        x = np.arange(len(MODELS)); w = 0.26
        for j, (ckey, clab) in enumerate(conds):
            vals, errs = [], []
            for m in MODELS:
                if ckey == "preword" and ks[m] and l in ks[m]["by_lang"]:
                    vals.append(ks[m]["by_lang"][l]["preword_bits"]["mean"]); errs.append(ks[m]["by_lang"][l]["preword_bits"]["sd"])
                elif ckey == "random_context" and ks[m] and l in ks[m]["by_lang"]:
                    vals.append(ks[m]["by_lang"][l]["random_bits"]["mean"]); errs.append(ks[m]["by_lang"][l]["random_bits"]["sd"])
                else:
                    vals.append(ko[m][l][ckey]["mean_delta_bits"]); errs.append(0)
            col = [BLUE, YELLOW, AQUA][j]
            ax.bar(x + (j - 1) * w, vals, w, yerr=errs, capsize=2.5, label=clab, color=col, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels([MNAME[m] for m in MODELS], rotation=12, fontsize=8)
        ax.set_title(LNAME[l], fontsize=10)
        ax.axhline(0, color=INK2, lw=0.8)
    axes[0].set_ylabel("Δ continuation log-prob (bits)")
    axes[0].legend(frameon=False, fontsize=8, title="blocked position", title_fontsize=8)
    fig.suptitle("Blocking attention to the pre-word position is uniquely damaging (±1 SD over 3 seeds)", fontsize=10, y=1.02)
    save(fig, "A6_knockout")


# ---- A7: tokens per forward by drafter with CI error bars (1.7B) ----
def a7_tokens_per_forward():
    m = "Qwen/Qwen3-1.7B-Base"
    sig = load(m, "significance")["decoding"]
    order = ["dictionary", "preword_head", "current_head", "ahead_heads", "word_head_then_ahead"]
    labels = ["dict.", "pre-word\nhead", "current\nhead", "ahead\nheads", "word+\nahead"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    for ax, l in zip(axes, ("ta", "hi")):
        d = sig[l]
        means = [d[k]["tokens_per_forward_ci"][0] for k in order]
        los = [d[k]["tokens_per_forward_ci"][0] - d[k]["tokens_per_forward_ci"][1] for k in order]
        his = [d[k]["tokens_per_forward_ci"][2] - d[k]["tokens_per_forward_ci"][0] for k in order]
        x = np.arange(len(order))
        ax.bar(x, means, 0.6, yerr=[los, his], capsize=3, color=BLUE, edgecolor="white")
        ax.axhline(1.0, color=INK2, ls="--", lw=0.9)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_title(LNAME[l], fontsize=10)
    axes[0].set_ylabel("Tokens per forward pass")
    axes[0].text(0, 1.02, "plain = 1.0", fontsize=7.5, color=INK2)
    fig.suptitle("Lossless tokens-per-forward on Qwen3-1.7B (95% bootstrap CI over prompts)", fontsize=10, y=1.02)
    save(fig, "A7_tokens_per_forward")


# ---- A8: forest plot of effect sizes with CIs ----
def a8_forest():
    rows = []  # (label, mean, lo, hi, panel)
    for m in MODELS:
        sig = load(m, "significance")
        for l in ("ta", "hi"):
            d = sig["patching"][l]
            rows.append((f"{MNAME[m]} · {LNAME[l]}", d["transfer_diff_ci"][0], d["transfer_diff_ci"][1], d["transfer_diff_ci"][2], 0))
            k = sig["knockout"][l]
            rows.append((f"{MNAME[m]} · {LNAME[l]}", -k["preword_vs_random_bits_ci"][0], -k["preword_vs_random_bits_ci"][2], -k["preword_vs_random_bits_ci"][1], 1))
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.8), sharey=True)
    titles = ["Patching: first-token − pre-word\ntransfer rate", "Knockout: pre-word cost over random\n(bits, positive = more damage)"]
    for panel, (ax, title) in enumerate(zip(axes, titles)):
        pr = [r for r in rows if r[4] == panel]
        y = np.arange(len(pr))[::-1]
        for yi, (lab, mu, lo, hi, _) in zip(y, pr):
            ax.plot([lo, hi], [yi, yi], color=INK2, lw=1.4)
            ax.plot(mu, yi, "o", color=ORANGE if panel else BLUE, ms=6, markeredgecolor=INK, zorder=3)
        ax.set_yticks(y)
        if panel == 0:
            ax.set_yticklabels([r[0] for r in pr], fontsize=7.5)
        ax.axvline(0, color="#b23b3b", ls="--", lw=0.9)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", visible=False)
        ax.margins(y=0.08)
    fig.suptitle("Effect sizes with 95% bootstrap CIs (all exclude zero)", fontsize=10.5, y=1.03)
    save(fig, "A8_forest")


if __name__ == "__main__":
    a1_first_token_share()
    a2_fragmentation_law()
    a3_speedup_vs_fragmentation()
    a4_patching_heatmap()
    a5_probe_by_layer()
    a6_knockout()
    a7_tokens_per_forward()
    a8_forest()
    print("charts done")


# ===================== diagrams (B1-B5) =====================
def _ax(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off"); ax.grid(False)
    return fig, ax


def box(ax, x, y, w, h, text, fc=SURF, ec=INK2, fs=8.5, tc=INK, lw=1.1, style="round,pad=0.02,rounding_size=2"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, mutation_scale=1)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, zorder=5)
    return (x + w / 2, y + h / 2), (x, y, w, h)


def arrow(ax, p1, p2, color=INK, lw=1.4, style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=12, color=color, lw=lw, shrinkA=2, shrinkB=2, zorder=1))


# ---- B1: system pipeline ----
def b1_pipeline():
    fig, ax = _ax((8.2, 4.2))
    c1, _ = box(ax, 2, 58, 18, 20, "Wikipedia\nTamil / Hindi /\nEnglish", fc="#e8eff8", ec=BLUE)
    c2, _ = box(ax, 28, 58, 20, 20, "Frozen LM\n(teacher-forced\nforward pass)", fc="#e8eff8", ec=BLUE)
    c3, _ = box(ax, 56, 58, 22, 20, "Cache per token:\nlog-prob, entropy,\nhidden states", fc="#e8eff8", ec=BLUE)
    arrow(ax, (20, 68), (28, 68)); arrow(ax, (48, 68), (56, 68))
    outs = ["Spelling share\n& boundaries (§IV)", "Plan probe (§V)",
            "Activation patching (§VI)", "Attention knockout (§VI)", "Lossless decoder (§VII)"]
    cols = [AQUA, AQUA, ORANGE, ORANGE, YELLOW]
    for i, (t, col) in enumerate(zip(outs, cols)):
        yy = 44 - i * 9.2
        cc, _ = box(ax, 70, yy, 26, 7, t, fc="white", ec=col, fs=8)
        arrow(ax, (67, 60), (70, yy + 3.5), color=col, lw=1.1)
    box(ax, 56, 50, 22, 0.1, "", fc="none", ec="none")
    ax.plot([67, 67], [12, 60], color=INK2, lw=1.0)
    arrow(ax, (67, 68), (67, 60))
    ax.text(50, 92, "Figure B1.  One expensive corpus pass feeds every cheap downstream analysis",
            ha="center", fontsize=9.5, color=INK)
    save(fig, "B1_pipeline")


# ---- B2: word unit and boundary types ----
def b2_boundary():
    fig, ax = _ax((8.0, 3.2))
    xs = [8, 24, 40, 56, 72]
    labels = ["t₁", "t₂", "t₃", "t₄", "t₅"]
    sub = ["shared\nscript bytes", "", "", "", ""]
    for i, (x, lab) in enumerate(zip(xs, labels)):
        fc = "#f3d9cf" if i == 0 else "#e8eff8"
        ec = ORANGE if i == 0 else BLUE
        box(ax, x, 50, 14, 16, lab, fc=fc, ec=ec, fs=12)
        if sub[i]:
            ax.text(x + 7, 44, sub[i], ha="center", va="top", fontsize=7, color=ORANGE)
    btypes = ["byte", "grapheme", "byte", "subword"]
    for i, bt in enumerate(btypes):
        xb = (xs[i] + 14 + xs[i + 1]) / 2
        ax.annotate("", xy=(xb + 1.5, 58), xytext=(xb - 1.5, 58),
                    arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.0))
        ax.text(xb, 69, bt, ha="center", fontsize=7.5, color=INK2, rotation=0)
    # word bracket
    ax.annotate("", xy=(86, 42), xytext=(8, 42), arrowprops=dict(arrowstyle="-", color=INK, lw=1.0))
    ax.text(47, 36, "one word  (first token ≈ 0 bits;  information spread across continuation tokens)",
            ha="center", fontsize=8.5, color=INK)
    ax.text(50, 86, "Figure B2.  A fragmented Indic word: first token is shared, boundaries are byte / grapheme / subword",
            ha="center", fontsize=9, color=INK)
    save(fig, "B2_boundary")


# ---- B3: three interventions ----
def b3_interventions():
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0))
    for ax in axes:
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off"); ax.grid(False)

    def strip(ax, hi_pre=False, hi_first=False):
        labs = ["ctx", "ctx", "pre", "t₁", "t₂", "t₃"]
        for i, lb in enumerate(labs):
            x = 6 + i * 15
            fc = "#e8eff8"
            if hi_pre and lb == "pre":
                fc = "#f3d9cf"
            if hi_first and lb == "t₁":
                fc = "#d7f0e6"
            box(ax, x, 38, 13, 14, lb, fc=fc, ec=INK2, fs=8)
        return

    # probe
    strip(axes[0], hi_pre=True)
    arrow(axes[0], (6 + 2 * 15 + 6.5, 54), (6 + 2 * 15 + 6.5, 74), color=BLUE)
    axes[0].text(50, 82, "Probe: read word identity\nfrom the pre-word state", ha="center", fontsize=8.5, color=BLUE)
    axes[0].set_title("(a) probe — is it readable?", fontsize=9)
    # patching
    strip(axes[1], hi_pre=True)
    axes[1].annotate("", xy=(6 + 2 * 15 + 6.5, 56), xytext=(6 + 2 * 15 + 6.5, 76),
                     arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.6))
    axes[1].text(50, 84, "Patch: swap pre-word state\nfrom another context", ha="center", fontsize=8.5, color=ORANGE)
    axes[1].set_title("(b) patching — is it stored here?", fontsize=9)
    # knockout
    strip(axes[2], hi_pre=True)
    p_pre = (6 + 2 * 15 + 6.5, 52); p_w = (6 + 3 * 15 + 6.5, 52)
    axes[2].annotate("", xy=p_pre, xytext=p_w, arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.3,
                     connectionstyle="arc3,rad=-0.5"))
    axes[2].text(6 + 2.6 * 15, 70, "✕", ha="center", fontsize=16, color="#c0392b", fontweight="bold")
    axes[2].text(50, 84, "Knockout: block attention\nto the pre-word position", ha="center", fontsize=8.5, color="#c0392b")
    axes[2].set_title("(c) knockout — is it needed?", fontsize=9)
    fig.suptitle("Figure B3.  Triangulating the pre-word position: readable, necessary, but not localized", fontsize=9.5, y=1.04)
    save(fig, "B3_interventions")


# ---- B4: lossless decoder flow ----
def b4_decoder_flow():
    fig, ax = _ax((5.2, 6.6))
    a, _ = box(ax, 25, 88, 50, 8, "target greedy token  t", fc="#e8eff8", ec=BLUE)
    b, bb = box(ax, 20, 72, 60, 9, "t starts a new word?", fc="white", ec=INK2)  # decision
    c, _ = box(ax, 18, 56, 64, 9, "drafter proposes  d = (d₁…dₖ)\nfrom current hidden state", fc="#fff4e0", ec=YELLOW)
    d, _ = box(ax, 20, 42, 60, 8, "ONE forward pass on  [t, d₁…dₖ]", fc="#e8eff8", ec=BLUE)
    e, _ = box(ax, 16, 26, 68, 9, "accept dᵢ while dᵢ = model's own\ngreedy arg max  (j tokens accepted)", fc="#d7f0e6", ec=AQUA)
    f, _ = box(ax, 20, 12, 60, 8, "crop KV cache to last accepted", fc="#f3d9cf", ec=ORANGE)
    g, _ = box(ax, 28, 0, 44, 7, "output ≡ plain greedy  (lossless)", fc="white", ec=INK, fs=8.5)
    arrow(ax, (50, 88), (50, 81))
    arrow(ax, (50, 72), (50, 65)); ax.text(52, 68, "yes", fontsize=7.5, color=INK2)
    arrow(ax, (50, 56), (50, 50)); arrow(ax, (50, 42), (50, 35)); arrow(ax, (50, 26), (50, 20))
    arrow(ax, (50, 12), (50, 7))
    # no branch
    arrow(ax, (80, 76.5), (90, 76.5)); ax.text(84, 79, "no", fontsize=7.5, color=INK2)
    ax.plot([90, 90], [76.5, 30], color=INK2, lw=1.0); arrow(ax, (90, 30), (84, 30))
    ax.text(50, 98, "Figure B4.  Draft–verify loop; acceptance is defined by the\ntarget's own arg max, so output is provably unchanged", ha="center", fontsize=8.5, color=INK)
    save(fig, "B4_decoder_flow")


# ---- B5: drafter head architectures ----
def b5_heads():
    fig, ax = _ax((8.2, 3.6))
    # word head (top row)
    ax.text(3, 86, "Word head", fontsize=9.5, color=INK, fontweight="bold")
    wy = 66
    p1, _ = box(ax, 3, wy, 15, 12, "hidden\nstate  h", fc="#e8eff8", ec=BLUE, fs=8)
    p2, _ = box(ax, 24, wy, 16, 12, "standardize\n(μ, σ)", fc="white", ec=INK2, fs=8)
    p3, _ = box(ax, 46, wy, 16, 12, "linear\nW h + b", fc="white", ec=INK2, fs=8)
    p4, _ = box(ax, 68, wy, 28, 12, "softmax over\nlexicon ∪ {OTHER}", fc="#d7f0e6", ec=AQUA, fs=8)
    for a, b in [(p1, p2), (p2, p3), (p3, p4)]:
        arrow(ax, (a[0] + 8, a[1]), (b[0] - 8, b[1]))
    # token-ahead head (bottom row)
    ax.text(3, 40, "Token-ahead heads (Medusa-style)", fontsize=9.5, color=INK, fontweight="bold")
    ty = 18
    q1, _ = box(ax, 3, ty, 20, 12, "h  and  E[token t]", fc="#e8eff8", ec=BLUE, fs=8)
    q2, _ = box(ax, 30, ty, 20, 12, "K parallel\nlinear heads", fc="#fff4e0", ec=YELLOW, fs=8)
    q3, _ = box(ax, 57, ty, 39, 12, "P(t+1), …, P(t+K)\nover small vocab ∪ {STOP}", fc="#d7f0e6", ec=AQUA, fs=8)
    for a, b in [(q1, q2), (q2, q3)]:
        arrow(ax, (a[0] + 10, a[1]), (b[0] - 10, b[1]))
    ax.text(50, 96, "Figure B5.  Both drafters are linear, fit by convex optimization on frozen features", ha="center", fontsize=9, color=INK)
    save(fig, "B5_heads")


def diagrams():
    b1_pipeline(); b2_boundary(); b3_interventions(); b4_decoder_flow(); b5_heads()
    print("diagrams done")


if __name__ == "__main__":
    diagrams()
