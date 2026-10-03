"""Map subword tokens to words, and classify every within-word token boundary.

A word is a maximal run of tokens that starts at a whitespace or punctuation
boundary. Each continuation boundary (between two tokens of the same word) is
classified by *why* the tokenizer split there:

* ``byte``     - the next token starts in the middle of a UTF-8 code point
                 (pure byte-level BPE artifact; Tamil characters are 3 bytes)
* ``grapheme`` - the next token starts with a combining mark (vowel sign,
                 virama, ZWJ/ZWNJ), i.e. inside one written syllable
* ``subword``  - a split between whole graphemes (a "real" subword split)

This decomposition lets the analysis separate encoding artifacts from genuine
word-internal decisions.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

WHITESPACE = {0x20, 0x0A, 0x09, 0x0D}
JOINERS = {"‌", "‍"}


def _bytes_to_unicode() -> dict[int, str]:
    # Same table as GPT-2 / tiktoken byte-level BPE.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


_BYTE_DECODER = {v: k for k, v in _bytes_to_unicode().items()}


def _sp_piece_bytes(piece: str) -> bytes:
    if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
        return bytes([int(piece[3:5], 16)])
    return piece.replace("▁", " ").encode("utf-8")


def token_bytes_table(tokenizer) -> list[bytes]:
    """Raw bytes for every token id (byte-level BPE or SentencePiece)."""
    vocab_size = len(tokenizer)
    pieces = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
    special = set(getattr(tokenizer, "all_special_tokens", []))
    added = {t.content if hasattr(t, "content") else str(t) for t in getattr(tokenizer, "added_tokens_decoder", {}).values()}
    # Decide the scheme from the space marker that dominates the vocab.
    n_sp = sum(1 for p in pieces[:5000] if p and "▁" in p)
    n_bl = sum(1 for p in pieces[:5000] if p and "Ġ" in p)
    sentencepiece = n_sp > n_bl
    table: list[bytes] = []
    for p in pieces:
        if p is None:
            table.append(b"")
        elif p in special or p in added:
            table.append(p.encode("utf-8"))
        elif sentencepiece:
            table.append(_sp_piece_bytes(p))
        elif all(ch in _BYTE_DECODER for ch in p):
            table.append(bytes(_BYTE_DECODER[ch] for ch in p))
        else:
            table.append(p.encode("utf-8"))
    return table


def _first_char(b: bytes) -> str | None:
    """First complete character of ``b`` if ``b`` starts at a code-point boundary."""
    if not b or (b[0] & 0xC0) == 0x80:
        return None
    for k in range(1, 5):
        try:
            return b[:k].decode("utf-8")[0]
        except UnicodeDecodeError:
            continue
    return None


def _last_char(b: bytes) -> str | None:
    for k in range(1, 5):
        if k > len(b):
            break
        try:
            return b[-k:].decode("utf-8")[-1]
        except UnicodeDecodeError:
            continue
    return None


def _is_letterlike(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat[0] in ("L", "M") or ch in JOINERS


def _is_break_char(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat[0] in ("P", "S", "Z", "C") and ch not in JOINERS


@dataclass
class Unit:
    start: int  # token index (inclusive)
    end: int  # token index (exclusive)
    kind: str  # word | number | punct | other
    text: str
    ids: tuple[int, ...]
    boundaries: list[str] = field(default_factory=list)  # len = n_tokens - 1

    @property
    def n_tokens(self) -> int:
        return self.end - self.start


class Segmenter:
    """Segments token-id sequences into word units for a given tokenizer."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.table = token_bytes_table(tokenizer)

    @lru_cache(maxsize=None)
    def _starts_unit(self, tid: int) -> bool:
        b = self.table[tid]
        if not b:
            return True
        if b[0] in WHITESPACE:
            return True
        ch = _first_char(b)
        return ch is not None and _is_break_char(ch)

    @lru_cache(maxsize=None)
    def _ends_with_break(self, tid: int) -> bool:
        b = self.table[tid]
        if not b:
            return True
        ch = _last_char(b)
        return ch is not None and _is_break_char(ch) and not (b[-1] & 0x80)

    def boundary_kind(self, prev_bytes: bytes, tid: int) -> str:
        b = self.table[tid]
        if b and (b[0] & 0xC0) == 0x80:
            return "byte"
        ch = _first_char(b)
        if ch is not None and (unicodedata.category(ch).startswith("M") or ch in JOINERS):
            return "grapheme"
        # previous token ended inside a code point -> also a byte split
        try:
            prev_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return "byte"
        return "subword"

    def segment(self, ids: list[int]) -> list[Unit]:
        units: list[Unit] = []
        start = 0
        for j in range(1, len(ids) + 1):
            if j == len(ids) or self._starts_unit(ids[j]) or self._ends_with_break(ids[j - 1]):
                units.append(self._make_unit(ids, start, j))
                start = j
        return units

    def _make_unit(self, ids: list[int], s: int, e: int) -> Unit:
        raw = b"".join(self.table[i] for i in ids[s:e])
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            kind = "other"
        elif all(_is_letterlike(c) for c in text):
            kind = "word"
        elif any(c.isdigit() for c in text) and all(c.isdigit() or c in ".,٫" for c in text):
            kind = "number"
        elif all(_is_break_char(c) for c in text):
            kind = "punct"
        else:
            kind = "other"
        bounds = []
        acc = self.table[ids[s]]
        for j in range(s + 1, e):
            bounds.append(self.boundary_kind(acc, ids[j]))
            acc += self.table[ids[j]]
        return Unit(start=s, end=e, kind=kind, text=text, ids=tuple(ids[s:e]), boundaries=bounds)
