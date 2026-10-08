"""TokenLift public API: load a target model, fit/attach a word-scoped EAGLE
head, and generate with the lossless word-level speculative decoder."""
from __future__ import annotations

import torch

from wordhead.models import load as _load
from wordhead.words import Segmenter
from wordhead.specdec import generate as _generate
from wordhead.sampling import generate_sample as _generate_sample
from wordhead.eagle import (WordEagleDrafter, WordEagleHead, ahead_vocab,
                            eagle_seq_data, fit_word_eagle_rollout)


class TokenLift:
    def __init__(self, lm):
        self.lm = lm
        self.seg = Segmenter(lm.tokenizer)
        self.emb = lm.model.get_input_embeddings().weight
        self.heads: dict[str, WordEagleHead] = {}
        self.vocabs: dict[str, list[int]] = {}
        self._drafters: dict[str, WordEagleDrafter] = {}

    # ---------- construction ----------
    @classmethod
    def from_pretrained(cls, model_name: str, device: str = "auto", dtype: str = "float32"):
        """Load a target model.

        ``dtype`` defaults to ``"float32"`` so that lossless greedy decoding is
        bit-identical to the model's own greedy output. On Apple-GPU (MPS) and
        other half-precision paths, near-tied logits can flip with batch shape,
        so a fast run may differ from a plain run at a tie; this is a property
        of float arithmetic, not of the method. Pass ``dtype="float16"`` (or
        ``"bfloat16"``) for speed when exact bit-identity is not required.
        """
        if dtype in ("float16", "bfloat16"):
            import warnings
            warnings.warn(
                f"TokenLift loaded in {dtype}: greedy output may differ from "
                "plain greedy decoding at floating-point ties. Use "
                'dtype="float32" for strict bit-identity.',
                stacklevel=2,
            )
        return cls(_load(model_name, device=device, dtype=dtype))

    def fit(self, langs, records_fn=None, vocab_size=4000, epochs=30, max_draft=8):
        """Train a word-scoped EAGLE head per language on cached corpus features.

        records_fn(lang) -> list[SeqRecord] with pre-word/in-word hidden states
        (see wordhead.corpus.run_corpus). Defaults to the research cache loader.
        """
        if records_fn is None:
            from experiments.common import load_records
            model = self.lm.name
            records_fn = lambda lg: load_records(model, lg, "fit")
        last = self.lm.n_layers
        for lang in langs:
            recs = records_fn(lang)
            vocab = ahead_vocab(recs, size=vocab_size)
            ST, TOK, Y, TM, FM = eagle_seq_data(recs, last, vocab, K=6)
            head = fit_word_eagle_rollout(ST, TOK, Y, TM, FM, self.emb, len(vocab), epochs=epochs)
            self._attach(lang, head, vocab, max_draft)
        return self

    def save_head(self, lang, path):
        torch.save({"state": self.heads[lang].state_dict(), "vocab": self.vocabs[lang]}, path)

    def load_head(self, lang, path, vocab=None, max_draft=8):
        """Load a trained head. Accepts either a dict {"state","vocab"} saved by
        save_head, or a bare state_dict saved by experiments/11_eagle.py (in which
        case pass the matching `vocab`, or let it be rebuilt from the fit cache).
        Layer dims are inferred from the checkpoint tensors."""
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "state" in ckpt:
            state, vocab = ckpt["state"], ckpt.get("vocab", vocab)
        else:
            state = ckpt
        if vocab is None:
            from experiments.common import load_records
            vocab = ahead_vocab(load_records(self.lm.name, lang, "fit"), size=state["to_tok.weight"].shape[0] - 2)
        hidden = state["inp.weight"].shape[0]
        n_vocab = state["to_tok.weight"].shape[0] - 2
        assert len(vocab) == n_vocab, f"vocab size {len(vocab)} != head n_vocab {n_vocab}"
        head = WordEagleHead(self.lm.d_model, self.emb.shape[1], n_vocab, hidden)
        head.load_state_dict(state)
        head.eval()
        self._attach(lang, head, vocab, max_draft)
        return self

    def _attach(self, lang, head, vocab, max_draft):
        self.heads[lang] = head
        self.vocabs[lang] = vocab
        self._drafters[lang] = WordEagleDrafter(head, self.emb, vocab, max_draft=max_draft)

    # ---------- generation ----------
    def _ids(self, prompt):
        return self.lm.tokenizer(prompt, add_special_tokens=False)["input_ids"] if isinstance(prompt, str) else list(prompt)

    def generate(self, prompt, max_new=64, lang=None, accelerate=True):
        """Lossless greedy decoding. accelerate=False runs plain decoding."""
        ids = self._ids(prompt)
        drafter = self._pick(lang) if accelerate else None
        r = _generate(self.lm, ids, max_new, drafter=drafter, layer=self.lm.n_layers, segmenter=self.seg)
        return dict(text=self.lm.tokenizer.decode(r.tokens), tokens=r.tokens,
                    tokens_per_forward=r.tokens_per_forward, n_forward=r.n_forward)

    def sample(self, prompt, max_new=64, temperature=0.8, lang=None, seed=0):
        """Lossless (distribution-preserving) speculative sampling."""
        ids = self._ids(prompt)
        lang = lang or next(iter(self.heads))
        r = _generate_sample(self.lm, ids, max_new, self.heads[lang], self.emb,
                             temperature=temperature, segmenter=self.seg, layer=self.lm.n_layers, seed=seed)
        acc = sum(r.accepted) / max(sum(r.draft_lengths), 1)
        return dict(text=self.lm.tokenizer.decode(r.tokens), tokens=r.tokens, n_forward=r.n_forward,
                    tokens_per_forward=r.tokens_per_forward, acceptance=acc)

    def _pick(self, lang):
        if not self._drafters:
            raise RuntimeError("No head attached; call fit(...) or load_head(...) first.")
        return self._drafters[lang or next(iter(self._drafters))]
