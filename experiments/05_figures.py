"""Render every figure from results/<model>/*.json into figures/<model>/.

    python experiments/05_figures.py --models Qwen/Qwen3-0.6B-Base Qwen/Qwen3-1.7B-Base
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIGS, RESULTS, slug

# Validated categorical palette (light surface), fixed order; ink never uses series colors.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LANG = {"ta": "Tamil", "hi": "Hindi", "en": "English"}
LANG_COLOR = {"ta": BLUE, "hi": ORANGE, "en": AQUA}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
})


def load(model, name):
    p = RESULTS / slug(model) / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, model, name):
    d = FIGS / slug(model)
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", d / f"{name}.png")


def fig_spelling(model):
    d = load(model, "spelling_share")
    if not d:
        return
    langs = [l for l in ("ta", "hi", "en") if l in d]
    fig, ax = plt.subplots(figsize=(7, 0.7 * len(langs) + 1.5))
    ax.grid(axis="y", visible=False)
    parts = [("First token of a word", AQUA), ("Continuation, uncertain (p < 0.9)", ORANGE),
             ("Continuation, near-certain (p ≥ 0.9)", BLUE)]
    for i, l in enumerate(langs):
        s = d[l]
        vals = [1 - s["cont_share_of_word_tokens"], s["cont_share_of_word_tokens"] - s["spelling_share@0.9"],
                s["spelling_share@0.9"]]
        left = 0
        for (lab, col), v in zip(parts, vals):
            ax.barh(i, v, left=left, color=col, edgecolor=SURFACE, linewidth=2, height=0.55, label=lab if i == 0 else None)
            if v > 0.06:
                ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
            left += v
        ax.text(1.02, i, f"{s['tokens_per_word']:.1f} tokens/word", va="center", color=INK2, fontsize=9)
    ax.set_yticks(range(len(langs)), [LANG[l] for l in langs])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_xlabel("Share of tokens inside words")
    ax.set_title(f"What the model's word tokens are spent on ({slug(model)})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3, fontsize=8)
    save(fig, model, "fig1_spelling_share")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for l in langs:
        pos = [r for r in d[l]["by_position"] if r["n"] >= 200][:12]
        ax.plot([r["position"] + 1 for r in pos], [r["bits"] for r in pos], marker="o", color=LANG_COLOR[l], label=LANG[l])
    ax.set_xlabel("Token position inside the word")
    ax.set_ylabel("Surprisal (bits)")
    ax.set_title(f"Where a word's information sits ({slug(model)})")
    ax.legend()
    save(fig, model, "fig2_bits_by_position")


def fig_probe(model):
    d = load(model, "plan_probe")
    if not d:
        return
    langs = [l for l in ("ta", "hi") if l in d and d[l].get("by_layer")]
    if not langs:
        return
    fig, axes = plt.subplots(1, len(langs), figsize=(5.2 * len(langs), 3.6), sharey=True, squeeze=False)
    for ax, l in zip(axes[0], langs):
        r = d[l]["by_layer"]
        x = [a["layer"] for a in r]
        ax.plot(x, [a["acc_model_spells"] for a in r], marker="o", color=AQUA, label="Probe, words the model spells correctly")
        ax.plot(x, [a["acc"] for a in r], marker="o", color=BLUE, label="Probe, all words")
        ax.plot(x, [a["acc_control"] for a in r], marker="s", color=ORANGE, label="Control (shuffled labels)")
        ax.plot(x, [a["acc_majority"] for a in r], marker="^", color=INK2, linestyle="--", label="Most frequent word")
        ax.set_title(f"{LANG[l]}: {r[0]['n_classes']} candidate words")
        ax.set_xlabel("Layer read at the position before the word")
        ax.set_ylim(0, 1)
    axes[0][0].set_ylabel("Accuracy picking the upcoming word")
    axes[0][-1].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"The upcoming word is linearly readable before any of it is written ({slug(model)})", x=0.02, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, model, "fig3_plan_probe_by_layer")

    langs = [l for l in ("ta", "hi") if l in d and d[l].get("by_position")]
    if langs:
        fig, axes = plt.subplots(1, len(langs), figsize=(4.8 * len(langs), 3.6), sharey=True, squeeze=False)
        for ax, l in zip(axes[0], langs):
            r = d[l]["by_position"]
            x = [a["prefix_tokens"] for a in r]
            ax.plot(x, [a["acc_state"] for a in r], marker="o", color=BLUE, label="Probe on the model's state")
            ax.plot(x, [a["acc_prefix_only"] for a in r], marker="s", color=ORANGE, label="Written prefix only (dictionary)")
            ax.set_title(LANG[l])
            ax.set_xlabel("Tokens of the word already written")
            ax.set_xticks(x)
            ax.set_ylim(0, 1.02)
        axes[0][0].set_ylabel("Accuracy identifying the whole word")
        axes[0][0].legend(fontsize=8, loc="lower right")
        fig.suptitle(f"The model knows the word before the spelling gives it away ({slug(model)})", x=0.02, ha="left",
                     fontweight="bold", fontsize=11)
        fig.tight_layout()
        save(fig, model, "fig4_word_identity_as_word_unfolds")


def fig_patching(model):
    d = load(model, "patching")
    if not d:
        return
    langs = [l for l in ("ta", "hi", "en") if l in d]
    fig, axes = plt.subplots(1, len(langs), figsize=(5.4 * len(langs), 3.6), sharey=True, squeeze=False)
    for ax, l in zip(axes[0], langs):
        for where, col, lab in (("preword", BLUE, "Patch the position before the word"),
                                ("first_token", ORANGE, "Patch the word's first token (control)")):
            r = d[l][where]
            ax.fill_between(r["layers"], r["effect_q25"], r["effect_q75"], color=col, alpha=0.15, linewidth=0)
            ax.plot(r["layers"], r["effect_median"], marker="o", color=col, label=lab)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_title(f"{LANG[l]} ({d[l]['preword']['n_pairs']} context pairs)")
        ax.set_xlabel("Layer patched")
    axes[0][0].set_ylabel("Shift toward the donor word\n(0 = none, 1 = full)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"Where the choice of word is causally stored ({slug(model)})", x=0.02, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, model, "fig5_patching")


def fig_decoding(model):
    d = load(model, "heads_and_decoding")
    if not d:
        return
    langs = [l for l in ("ta", "hi", "en") if l in d]
    last = None
    rows = [("oracle", "Upper bound (model's own greedy)", INK2), ("dictionary", "Dictionary (context-blind)", ORANGE)]
    fig, axes = plt.subplots(1, len(langs), figsize=(4.6 * len(langs), 3.4), sharex=True, squeeze=False)
    for ax, l in zip(axes[0], langs):
        em = d[l]["emulation"]
        pre = sorted((k for k in em if k.startswith("preword_head_L")), key=lambda k: int(k.split("L")[-1]))[-1]
        cur = [k for k in em if k.startswith("current_head_L")][0]
        items = [("Dictionary (context-blind)", em["dictionary"], ORANGE), ("Word head, pre-word state", em[pre], AQUA),
                 ("Word head, current state", em[cur], BLUE), ("Upper bound", em["oracle"], INK2)]
        ax.grid(axis="y", visible=False)
        for i, (lab, e, col) in enumerate(items):
            ax.barh(i, e["step_reduction"], color=col, height=0.55)
            ax.text(e["step_reduction"] + 0.03, i, f"×{e['step_reduction']:.2f}", va="center", fontsize=9, color=INK)
        ax.axvline(1, color=INK2, linewidth=0.8)
        ax.set_yticks(range(len(items)), [i[0] for i in items] if ax is axes[0][0] else [""] * len(items))
        ax.invert_yaxis()
        ax.set_title(LANG[l])
        ax.set_xlabel("Word tokens per forward pass")
    fig.suptitle(f"Fewer sequential decoding steps on held-out text ({slug(model)})", x=0.02, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, model, "fig6_decoding_steps")


def fig_scale(models):
    data = {m: (load(m, "spelling_share"), load(m, "plan_probe"), load(m, "heads_and_decoding")) for m in models}
    data = {m: v for m, v in data.items() if all(v)}
    if len(data) < 2:
        return
    names = [slug(m).replace("-Base", "") for m in data]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    panels = [("Near-certain continuation tokens\n(share of word tokens, p ≥ 0.9)", lambda s, p, h, l: s[l]["spelling_share@0.9"]),
              ("Pre-word probe accuracy\n(best layer)", lambda s, p, h, l: max(a["acc"] for a in p[l]["by_layer"])),
              ("Word tokens per forward pass\n(current-state head)", lambda s, p, h, l: [v for k, v in h[l]["emulation"].items() if k.startswith("current_head")][0]["step_reduction"])]
    for ax, (title, fn) in zip(axes, panels):
        for l in ("ta", "hi"):
            ys = []
            for m, (s, p, h) in data.items():
                try:
                    ys.append(fn(s, p, h, l))
                except (KeyError, ValueError, IndexError):
                    ys.append(None)
            ax.plot(names, ys, marker="o", color=LANG_COLOR[l], label=LANG[l])
        ax.set_title(title, fontsize=10)
    axes[0].legend()
    fig.suptitle("How the findings change with model size", x=0.02, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    d = FIGS / "scale"
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / "fig7_scale.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", d / "fig7_scale.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen3-0.6B-Base"])
    args = ap.parse_args()
    for m in args.models:
        fig_spelling(m)
        fig_probe(m)
        fig_patching(m)
        fig_decoding(m)
    fig_scale(args.models)


if __name__ == "__main__":
    main()
