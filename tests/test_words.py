from wordhead.words import Segmenter, token_bytes_table


def test_bytes_roundtrip(tokenizer):
    table = token_bytes_table(tokenizer)
    for text in ["Hello world.", "குழந்தைகள் வீட்டிலிருந்து பள்ளிக்கு", "बच्चे घर से स्कूल"]:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        assert b"".join(table[i] for i in ids).decode("utf-8") == text


def test_units_cover_sequence_and_match_words(tokenizer):
    seg = Segmenter(tokenizer)
    text = "குழந்தைகள் வீட்டிலிருந்து பள்ளிக்கு நடந்து சென்றனர்."
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    units = seg.segment(ids)
    # contiguous, complete cover
    assert units[0].start == 0 and units[-1].end == len(ids)
    for a, b in zip(units, units[1:]):
        assert a.end == b.start
    words = [u.text for u in units if u.kind == "word"]
    assert words == text.rstrip(".").split()
    assert units[-1].kind == "punct"


def test_boundary_labels(tokenizer):
    seg = Segmenter(tokenizer)
    for text in ["வீட்டிலிருந்து", "स्कूल", "internationalization"]:
        ids = tokenizer(" " + text, add_special_tokens=False)["input_ids"]
        (u,) = [u for u in seg.segment(ids) if u.kind == "word"]
        assert len(u.boundaries) == u.n_tokens - 1
        assert set(u.boundaries) <= {"byte", "grapheme", "subword"}
    # a pure-ASCII word can never have byte or grapheme splits
    ids = tokenizer(" internationalization", add_special_tokens=False)["input_ids"]
    (u,) = seg.segment(ids)
    assert set(u.boundaries) <= {"subword"}


def test_english_punctuation_split(tokenizer):
    seg = Segmenter(tokenizer)
    ids = tokenizer("Hello, world (again).", add_special_tokens=False)["input_ids"]
    kinds = [(u.text, u.kind) for u in seg.segment(ids)]
    assert ("Hello", "word") in kinds and ("world", "word") in kinds and ("again", "word") in kinds
    assert all(k != "word" for t, k in kinds if t in {",", "(", ").", ")", "."})
