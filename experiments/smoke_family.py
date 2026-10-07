"""Smoke test a candidate second-family model before a full run."""
import sys
from wordhead.models import load, decoder_layers
from wordhead.words import Segmenter, token_bytes_table
from wordhead.corpus import run_corpus

name = sys.argv[1] if len(sys.argv) > 1 else "facebook/xglm-564M"
print("loading", name)
lm = load(name, device="auto", dtype="auto")
print("ok:", lm.name, "layers", lm.n_layers, "d_model", lm.d_model, "device", lm.device)
print("decoder_layers found:", len(decoder_layers(lm.model)))
seg = Segmenter(lm.tokenizer)
tbl = token_bytes_table(lm.tokenizer)
print("byte-table size:", len(tbl))
ta = "தமிழ்நாடு இந்தியாவின் தெற்கே அமைந்துள்ள ஒரு மாநிலமாகும்."
ids = lm.tokenizer(ta, add_special_tokens=False)["input_ids"]
units = seg.segment(ids)
print("tamil ids:", len(ids), "units:", len(units))
for u in units[:6]:
    print("   ", u.kind, u.n_tokens, repr(u.text))
recs = run_corpus(lm, [ta, "This is a short English test sentence."], max_len=64,
                  batch_size=2, hidden_layers=[0, lm.n_layers], inword_layers=[lm.n_layers],
                  inword_positions=4, progress=False)
print("corpus ok, records:", len(recs), "preword layers:", sorted(recs[0].preword_hidden))
print("SMOKE_OK")
