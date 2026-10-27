# -*- coding: utf-8 -*-
"""源文档加载、分块与分词,以及兼容用的检索入口。

分词是与语言无关的:ASCII 单词 + 中日韩单字/二元组,再叠加
:mod:`faithsift.lexicon` 里的中英共享概念词元。这样一条中文论断与一段
英文证据即使没有任何数字,也能在"讲的是同一件事"上重叠起来。

本模块只管数据与分词。排序打分的逻辑在 :mod:`faithsift.retrieval`
(BM25、数值不变性、向量信号)。:func:`retrieve` 保留为薄封装,
方便只想拿个 top-k 的调用方,循环里请改用 ``Retriever`` 以免重复建索引。
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .lexicon import concepts as extract_concepts
from .numbers import normalize_numerals

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\.\-_]*")
_CJK = re.compile(r"[一-鿿]")


def _clean_word(word: str) -> str:
    """小写化并去掉尾部标点:``'Grass.' -> 'grass'``(否则句末词永远匹配不上)。"""
    return word.lower().rstrip(".-_")

CHUNK_CHARS = 400
CHUNK_OVERLAP = 60


@dataclass
class Chunk:
    doc: str
    idx: int
    text: str
    tokens: set[str] = field(default_factory=set)
    tf: dict[str, int] = field(default_factory=dict)
    length: int = 0
    concepts: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.tf:
            seq = tokenize_seq(self.text)
            self.tf = dict(Counter(seq))
            self.length = len(seq)
        if not self.tokens:
            self.tokens = set(self.tf)
        if not self.concepts:
            self.concepts = frozenset(extract_concepts(self.text))


def tokenize_seq(text: str) -> list[str]:
    """返回词序列(保留重复,供词频/BM25 使用)。

    词元由四部分构成:小写英文单词、中日韩单字、中日韩相邻二元组,
    以及中英共享的概念词元(``茶树``/``tea`` -> ``tea``)。
    另外把中文数词归一化后的阿拉伯数字**追加**为独立词元,
    使 ``两千`` 与 ``2,000`` 能对上;原文分词结果保持不变,
    因此 ``一爆``(咖啡概念)不会被拆坏。
    """
    toks = [_clean_word(w) for w in _WORD.findall(text)]
    cjk = _CJK.findall(text)
    toks.extend(cjk)
    toks.extend(a + b for a, b in zip(cjk, cjk[1:]))
    toks.extend(extract_concepts(text))
    toks.extend(_NUM_TOKEN.findall(normalize_numerals(text)))
    return toks


def tokenize(text: str) -> set[str]:
    """唯一词元集合(公开 API,v0.1 起未变)。"""
    return set(tokenize_seq(text))


def load_sources(paths: Sequence[Path]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in paths:
        text = Path(p).read_text(encoding="utf-8")
        # 按段落切分(空行分隔),再对超长段落做滑窗切块
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
                chunks.append(Chunk(doc=p.name, idx=idx, text=piece))
                if start + CHUNK_CHARS >= len(para):
                    break
                start += CHUNK_CHARS - CHUNK_OVERLAP
    return chunks


_NUM_TOKEN = re.compile(r"\d+(?:\.\d+)?")


def distinctive_numbers(text: str) -> set[str]:
    """抽取可用来标识事实的数字(至少两位,或带小数点)。

    先做中文数词归一化(``两千`` -> ``2000``)并去掉千分位逗号
    (``2,000`` -> ``2000``),所以中英混写的语料能直接比对。
    单个数字("3 bar"、"5 percent")区分度太低,不参与判定。
    """
    return {n for n in _NUM_TOKEN.findall(normalize_numerals(text))
            if len(n) >= 2 or "." in n}


def retrieve(claim: str, chunks: Sequence[Chunk], k: int = 3,
             backend: str = "lexical") -> list[tuple[Chunk, float]]:
    """向后兼容的一次性 top-k 检索(内部会新建一个检索器)。

    Prefer :class:`faithsift.retrieval.Retriever` in loops: it indexes once.
    """
    from .retrieval import Retriever

    return Retriever(chunks, backend=backend).search(claim, k=k)


def discover_sources(src: Path) -> list[Path]:
    src = Path(src)
    if src.is_file():
        return [src]
    exts = ("*.md", "*.txt")
    files: list[Path] = []
    for e in exts:
        files.extend(sorted(src.rglob(e)))
    return files


__all__ = [
    "CHUNK_CHARS", "CHUNK_OVERLAP", "Chunk", "discover_sources",
    "distinctive_numbers", "load_sources", "retrieve", "tokenize",
    "tokenize_seq",
]
