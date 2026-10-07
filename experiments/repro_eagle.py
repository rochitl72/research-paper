import faulthandler, sys
from common import load_records
from wordhead.models import load
from wordhead.eagle import WordEagleDrafter, ahead_vocab, eagle_training_data, fit_word_eagle
from wordhead.heads import emulated_steps

lm = load("Qwen/Qwen3-0.6B-Base")
emb = lm.model.get_input_embeddings().weight
fit = load_records("Qwen/Qwen3-0.6B-Base", "ta", "fit")[:200]
ev = load_records("Qwen/Qwen3-0.6B-Base", "ta", "eval")[:8]
vocab = ahead_vocab(fit, size=4000)
Fin, Tin, Fout, Y = eagle_training_data(fit, 28, vocab, max_steps=6)
head = fit_word_eagle(Fin, Tin, Fout, Y, emb, len(vocab), epochs=3)
d = WordEagleDrafter(head, emb, vocab, max_draft=10)
print("built drafter; calling one draft...", flush=True)
r0 = ev[0]
row = 0
h = r0.state_for_prefix(28, row, 1)
print("one draft result:", d.draft([r0.units[r0.preword_units[row]].ids[0]], h, h), flush=True)
print("running emulated_steps on 8 records with a 25s watchdog...", flush=True)
faulthandler.dump_traceback_later(25, exit=True)
res = emulated_steps(ev, d, layer=28)
print("DONE", res["step_reduction"], flush=True)
