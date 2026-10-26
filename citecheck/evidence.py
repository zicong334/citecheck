# -*- coding: utf-8 -*-
"""Source loading, chunking and lexical retrieval (zero-dependency).

Tokenization is language-agnostic: ASCII word tokens + CJK character bigrams,
which works reasonably for both Chinese and English without segmenters.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\.\-]*")
_CJK = re.compile(r"[一-鿿]")

CHUNK_CHARS = 400
CHUNK_OVERLAP = 60


@dataclass
class Chunk:
    doc: str
    idx: int
    text: str
    tokens: set[str] = field(default_factory=set)


def tokenize(text: str) -> set[str]:
    toks = {w.lower() for w in _WORD.findall(text)}
    cjk = _CJK.findall(text)
    toks.update(a + b for a, b in zip(cjk, cjk[1:]))
    toks.update(cjk)
    return toks


def load_sources(paths: list[Path]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in paths:
        text = Path(p).read_text(encoding="utf-8")
        # paragraph-aware: split on blank lines, then window long paragraphs
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if not para:
                continue
            start = 0
            while True:
                piece = para[start:start + CHUNK_CHARS]
                if not piece:
                    break
                idx = len(chunks)
                chunks.append(Chunk(doc=p.name, idx=idx, text=piece,
                                    tokens=tokenize(piece)))
                if start + CHUNK_CHARS >= len(para):
                    break
                start += CHUNK_CHARS - CHUNK_OVERLAP
    return chunks


_NUM_TOKEN = re.compile(r"\d+(?:\.\d+)?")


def _distinctive_nums(text: str) -> set[str]:
    return {n for n in _NUM_TOKEN.findall(text) if len(n) >= 2 or "." in n}


def retrieve(claim: str, chunks: list[Chunk], k: int = 3) -> list[tuple[Chunk, float]]:
    """Lexical containment + language-agnostic numeric-overlap boost.

    Numbers survive translation: a zh claim about "196 至 205 摄氏度" must
    retrieve the en chunk containing 196/205 even with zero token overlap.
    """
    ct = tokenize(claim)
    cn = _distinctive_nums(claim)
    if not ct and not cn:
        return []
    scored = []
    for c in chunks:
        inter = len(ct & c.tokens) / len(ct) if ct else 0.0
        if cn:
            chunk_nums = _distinctive_nums(c.text)
            num_hit = len(cn & chunk_nums) / len(cn)
        else:
            num_hit = 0.0
        score = inter + 0.6 * num_hit
        if score > 0:
            scored.append((c, score))
    scored.sort(key=lambda x: -x[1])
    return scored[:k]


def discover_sources(src: Path) -> list[Path]:
    src = Path(src)
    if src.is_file():
        return [src]
    exts = ("*.md", "*.txt")
    files: list[Path] = []
    for e in exts:
        files.extend(sorted(src.rglob(e)))
    return files
