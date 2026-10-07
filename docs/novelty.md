# Prior-work / novelty check

Literature check done 2026-10-06 to position the paper honestly. Searched for
work on (1) reading/planning future tokens from hidden states, (2) speculative
/ multi-token decoding for multilingual and low-resource settings, (3)
tokenization fragmentation and fairness for Indic scripts. Findings and how we
differ.

## Closest prior work

- **Future Lens** (Pal, Sun, Yuan, Wallace, Bau, CoNLL 2023). Shows several
  *future* tokens can be anticipated from a *single* hidden state, in English,
  via learned decoders / causal interventions. This is the closest analogue to
  our plan probe.
  *How we differ:* we ask about a specific, linguistically meaningful unit the
  tokenizer has shattered — whether the identity of a whole multi-token **word**
  is decodable *before it is written*, in heavily fragmented Indic scripts — and
  we add proper control-task baselines and split accuracy by whether the model
  then spells the word correctly. Crucially we do not stop at "readable": we test
  causality.

- **Do Language Models Plan Ahead for Future Tokens?** (Wu, Morris, Levine,
  COLM 2024). Frames two hypotheses — *pre-caching* (the model deliberately
  computes future-useful features now) vs *breadcrumbs* (present-token features
  happen to help later) — and finds natural-language modeling is mostly
  breadcrumbs at small scale.
  *How we differ / connect:* our patching + knockout give a concrete,
  mechanistic instance of their dichotomy for word identity under
  fragmentation. The pre-word position is a *necessary relay* (knockout) whose
  content does *not* carry the word (patching) — i.e. breadcrumbs, not a stored
  plan. We give their abstract question a measurable answer for a real unit.

- **Towards Fast Multilingual LLM Inference: Speculative Decoding and
  Specialized Drafters** (Yi, Kim, Jeung, Chang, Yun, arXiv 2024). Speeds up
  multilingual inference with language-specific *draft models*.
  *How we differ:* we do not train a separate draft model. We target the
  *within-word* fragmentation directly, with cheap linear word / token-ahead
  heads on the target model's own frozen features (laptop-fittable), and we use
  the decoder as a measurement of the representation, not only a speedup.

## Supporting / background work (already cited)

- Tokenizer fairness and cost across languages: Petrov et al. (NeurIPS 2023),
  Ahia et al. (EMNLP 2023). We give the *internal* counterpart: fragmentation
  moves a word's information out of its first token.
- Speculative decoding foundations: Leviathan et al. (ICML 2023), Chen et al.
  (2023); head-based variants Medusa (Cai et al. 2024), EAGLE (Li et al. 2024),
  blockwise parallel (Stern et al. 2018), multi-token prediction (Gloeckle et
  al. 2024).
- Probing and control tasks: Alain & Bengio (2017), Hewitt & Liang (2019).
- Causal methods: ROME causal tracing (Meng et al. 2022), IOI path patching
  (Wang et al. 2023), path patching (Goldowsky-Dill et al. 2023), tuned lens
  (Belrose et al. 2023), transformer-circuits framework (Elhage et al. 2021).
- Also noted (Indic morphology-aware tokenization, e.g. MorphTok and related
  2025--2026 work): orthogonal direction — they *change* the tokenizer; we
  *measure the consequences* of the byte-level tokenizer that shipped models
  actually use, and add a second family (XGLM) whose SentencePiece tokenizer is
  far less fragmenting as a control.

## Novelty statement

To our knowledge this is the first work to:
1. quantify, per script, *where in a fragmented word* a language model's
   information about word identity lives (first-token information share:
   ~8% Tamil / ~27% Hindi / ~96% English, replicated across model sizes and
   contrasted against a less-fragmenting tokenizer family);
2. triangulate probe + activation patching + attention knockout to show the
   upcoming word is **readable but not localized** before it is written (a
   necessary relay, not a stored plan) — giving a concrete answer to the
   pre-caching-vs-breadcrumbs question for a real linguistic unit; and
3. build a **lossless, laptop-fittable word-level** speculative decoder from
   this analysis, reporting gains that grow with model size.

## Honest limits of the novelty

- The decoder is an application of known head-based speculative decoding; its
  novelty is the word-level framing and the laptop/frozen-feature setup, not the
  mechanism.
- "First work" claims above are to the best of a 2026 literature check; a
  camera-ready should re-run the search and add any concurrent Indic-tokenization
  interpretability work.
