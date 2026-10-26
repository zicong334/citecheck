# -*- coding: utf-8 -*-
"""Claim splitting: break an answer into sentence-level claims (zh + en)."""
from __future__ import annotations

import re

_SPLIT = re.compile(r"(?<=[。!?;！？;\n])\s*")
_MIN_LEN_CJK = 6
_MIN_LEN_LATIN = 15


def _worth(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    cjk = sum(1 for ch in stripped if "一" <= ch <= "鿿")
    if cjk >= 4:
        return len(stripped) >= _MIN_LEN_CJK
    return len(stripped) >= _MIN_LEN_LATIN and any(
        ch.isalnum() for ch in stripped)


def split_claims(answer: str) -> list[str]:
    """Sentence-level claims, order preserved, boilerplate filtered.

    Skips pure headings/markdown decoration and very short acknowledgements.
    """
    out: list[str] = []
    for raw in _SPLIT.split(answer):
        text = raw.strip()
        if not text:
            continue
        # strip markdown decoration but keep content
        text = re.sub(r"^#+\s*", "", text)
        text = re.sub(r"^[-*>\s]+", "", text)
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        if not _worth(text):
            continue
        out.append(text)
    return out
