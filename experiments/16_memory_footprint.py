"""G3.8b Memory footprint: TokenLift head vs a full draft model.

The honest advantage of the draft-model-free head is memory. We report the
deterministic, hardware-independent figure: parameter count and the resulting
fp16 weight memory for (a) the trained TokenLift head actually used on the 4B
target, and (b) the Qwen3-0.6B-Base model that standard self-speculation must
keep resident alongside the target. Reads saved heads + model configs; counts
params by loading the draft on CPU meta-free is avoided -- we read its config.
"""
import argparse, json, glob, os
import torch
from transformers import AutoConfig


def head_params(path):
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) and "state_dict" in sd else sd
    return sum(v.numel() for v in sd.values() if torch.is_tensor(v))


def model_params_from_config(name):
    """Parameter count from config (no weights loaded): embeddings + blocks + head."""
    c = AutoConfig.from_pretrained(name)
    V, d = c.vocab_size, c.hidden_size
    L = c.num_hidden_layers
    inter = c.intermediate_size
    kv = getattr(c, "num_key_value_heads", c.num_attention_heads)
    hd = getattr(c, "head_dim", d // c.num_attention_heads)
    q = d * (c.num_attention_heads * hd)
    k = d * (kv * hd); v = d * (kv * hd); o = (c.num_attention_heads * hd) * d
    attn = q + k + v + o
    mlp = 3 * d * inter          # gate, up, down (SwiGLU)
    per_block = attn + mlp + 2 * d  # + 2 RMSNorms
    emb = V * d
    tie = getattr(c, "tie_word_embeddings", True)
    total = emb + L * per_block + d + (0 if tie else V * d)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen3-4B-Base")
    ap.add_argument("--draft", default="Qwen/Qwen3-0.6B-Base")
    args = ap.parse_args()
    cache = os.path.expanduser(f"~/wordhead-research/cache/{args.target}")
    heads = sorted(glob.glob(f"{cache}/*_eagle_head.pt"))
    hp = {os.path.basename(h).split("_")[0]: head_params(h) for h in heads}
    hp_mean = sum(hp.values()) / max(len(hp), 1)
    draft_p = model_params_from_config(args.draft)
    tgt_p = model_params_from_config("Qwen/" + args.target)

    def mb(n):  # fp16
        return n * 2 / 1e6

    out = dict(head_params=hp, head_params_mean=hp_mean, draft_params=draft_p,
               target_params=tgt_p,
               head_MB_fp16=round(mb(hp_mean), 2), draft_MB_fp16=round(mb(draft_p), 1),
               ratio_draft_over_head=round(draft_p / hp_mean, 1),
               head_pct_of_target=round(100 * hp_mean / tgt_p, 3))
    print(json.dumps(out, indent=1))
    with open(f"results/{args.target}/memory_footprint.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nTokenLift head: {hp_mean/1e6:.2f} M params ({mb(hp_mean):.1f} MB fp16)")
    print(f"Self-spec draft model (0.6B): {draft_p/1e6:.0f} M params ({mb(draft_p)/1000:.2f} GB fp16)")
    print(f"=> draft model is {draft_p/hp_mean:.0f}x the head; head is {100*hp_mean/tgt_p:.2f}% of the 4B target")


if __name__ == "__main__":
    main()
