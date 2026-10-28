# -*- coding: utf-8 -*-
"""检索后端:BM25 词法、数值不变性、可选向量通道。

三种信号,可叠加使用:

1. **BM25** —— 建在与语言无关的词元上(ASCII 单词 + 中日韩单字/二元组 +
   双语概念词元)。取代 v0.1 的朴素包含度:那个版本忽略词稀有度和文档长度,
   短句与口水词会把排序带偏。
2. **数值重合** —— 数字不随翻译改变,所以"196 至 205 摄氏度"的中文论断
   仍能命中英文段落。权重由 ``numeric_weight`` 控制。
3. **向量余弦** —— 可选的嵌入通道,面向开放领域语料和重度同义改写。
   它只能被注入,永远不是必需品:没有向量索引时 :class:`Retriever` 照样能用。

``backend`` 选择混合方式:

``lexical``  BM25 + 数值          (默认,零依赖)
``hybrid``   BM25 + 数值 + 向量    (无向量索引时自动退回 lexical)
``dense``    向量余弦 + 数值       (无向量索引时自动退回 lexical)
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Protocol, Sequence

from .evidence import Chunk, distinctive_numbers, tokenize_seq
from .lexicon import concepts, entities


class DenseIndex(Protocol):
    """能把一条查询与各片段向量打分的对象都可以接进来。"""

    def cosine(self, query: str) -> list[float]: ...


@dataclass
class BM25Index:
    """基于已分词块的 Okapi BM25 索引。

    ``scores()`` 按最高分归一化到 0..1,便于与其他信号(本身就在 0..1)相加;
    ``raw_scores()`` 保留原始值,供阈值判断与调试使用。
    ``idf`` 公开暴露:判官的加权覆盖率也复用这张表。
    """

    chunks: Sequence[Chunk]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.chunks = list(self.chunks)
        n = len(self.chunks)
        df: Counter[str] = Counter()
        for c in self.chunks:
            df.update(c.tf.keys())
        self.idf: dict[str, float] = {
            t: math.log(1.0 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()
        }
        self.avgdl = (sum(c.length for c in self.chunks) / n) if n else 0.0

    def raw_scores(self, query_tokens: Sequence[str]) -> list[float]:
        q = set(query_tokens)
        avgdl = self.avgdl or 1.0
        out: list[float] = []
        for c in self.chunks:
            s = 0.0
            norm = self.k1 * (1.0 - self.b + self.b * (c.length / avgdl))
            for t in q:
                f = c.tf.get(t)
                if not f:
                    continue
                s += self.idf.get(t, 0.0) * (f * (self.k1 + 1.0)) / (f + norm)
            out.append(s)
        return out

    def scores(self, query_tokens: Sequence[str]) -> list[float]:
        raw = self.raw_scores(query_tokens)
        top = max(raw) if raw else 0.0
        if top <= 0:
            return [0.0] * len(raw)
        return [r / top for r in raw]


def numeric_hit(claim: str, chunk: Chunk) -> float:
    """论断的特征数字有多大比例出现在 *chunk* 里。"""
    want = distinctive_numbers(claim)
    if not want:
        return 0.0
    have = distinctive_numbers(chunk.text)
    return len(want & have) / len(want)


class Retriever:
    """把词法、数值与可选向量信号合成为一个排名。"""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        backend: str = "lexical",
        numeric_weight: float = 0.6,
        dense: DenseIndex | None = None,
        dense_weight: float = 0.5,
        entity_filter: bool = False,
        concept_weight: float = 1.0,
    ) -> None:
        if backend not in ("lexical", "hybrid", "dense"):
            raise ValueError(f"unknown backend {backend!r}")
        self.chunks = list(chunks)
        self.backend = backend if (backend == "lexical" or dense is not None) else "lexical"
        self.numeric_weight = numeric_weight
        self.concept_weight = concept_weight
        self.dense = dense
        self.dense_weight = dense_weight
        # 实体过滤默认关闭:实体一致性由判官在 primary/others 分区里处理,
        # 检索层硬性排后会把"跨实体多跳"需要的另一半证据挤出 top-k。
        self.entity_filter = entity_filter
        self.bm25 = BM25Index(self.chunks)
        self._chunk_entities: list[set[str]] = (
            [entities(c.text) for c in self.chunks] if entity_filter else []
        )

    # -- scoring ---------------------------------------------------------
    def _base(self, claim: str) -> list[float]:
        qt = tokenize_seq(claim)
        lexical = self.bm25.scores(qt)
        if self.backend == "dense" and self.dense is not None:
            return _minmax(self.dense.cosine(claim))
        if self.backend == "hybrid" and self.dense is not None:
            dense = _minmax(self.dense.cosine(claim))
            return [(1.0 - self.dense_weight) * lexical[i] +
                    self.dense_weight * dense[i] for i in range(len(lexical))]
        return lexical

    def score_all(self, claim: str) -> list[float]:
        """综合得分 = 词法/向量基础分 + 概念重合 + 数值重合。

        加入概念重合的动机:中文论断的中文字/二元组数量远多于它的概念词元数,
        BM25 会把"共享很多中文二元组但并不在讲同一件事"的块排到前面。
        概念重合(``copenhagen``、``first_crack`` 这类共享词元)才是跨语言
        真正可靠的信号,因此给它与基础分同量级的权重。
        """
        base = self._base(claim)
        want = concepts(claim)
        out = []
        for i, c in enumerate(self.chunks):
            sem = (len(want & c.concepts) / len(want)) if want else 0.0
            out.append(base[i]
                       + self.concept_weight * sem
                       + self.numeric_weight * numeric_hit(claim, c))
        return out

    def search(self, claim: str, k: int = 3) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        scores = self.score_all(claim)
        order = sorted(range(len(self.chunks)), key=lambda i: -scores[i])
        order = self._prefer_claim_entities(claim, order, scores) \
            if self.entity_filter else order
        return [(self.chunks[i], round(scores[i], 4)) for i in order[:k]
                if scores[i] > 0]

    def _prefer_claim_entities(
        self, claim: str, order: list[int], scores: list[float]
    ) -> list[int]:
        """把"在讲别的实体"的片段降权。

        论断点名一个实体时,讲另一个实体的段落不能充当它的证据
        (阿姆斯特丹 vs 哥本哈根)。没提任何实体、或提到论断实体的片段保留;
        其余排到后面——只是排后,不丢弃,所以名录没覆盖时行为退化回普通 BM25。
        """
        want = entities(claim)
        if not want:
            return order
        keep, demote = [], []
        for i in order:
            mention = self._chunk_entities[i] if i < len(self._chunk_entities) else set()
            if mention and not (mention & want):
                demote.append(i)
            else:
                keep.append(i)
        return keep + demote


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def build_retriever(
    chunks: Sequence[Chunk],
    backend: str = "lexical",
    embedder_model: str | None = None,
    cache_dir: str | None = None,
    **kwargs,
) -> Retriever:
    """便捷工厂:传入 ``embedder_model`` 时顺带建好向量索引。

    ``embedder_model`` 取值:
      * ``hashing``            —— 零依赖离线替身(不跨语言),用于跑通接线;
      * ``local`` / ``local:<hf-model-id>`` —— 本地离线 ONNX 多语模型(fastembed);
      * 其它字符串             —— 视作 OpenAI 兼容端点的嵌入模型名(需要 key)。

    本地后端缺依赖或模型加载失败时,打印中文提示并**自动退回词法通道**
    (dense=None -> Retriever 把 backend 降级为 lexical),不会崩;这与
    "hybrid 未配 embedder 时退回 lexical" 是同一套降级哲学。
    """
    dense = None
    if embedder_model:
        from .embed import (CachedEmbedder, DenseIndex as _DenseIndex,
                            HashingEmbedder, SemanticBackendUnavailable)

        emb = None
        if embedder_model == "hashing":
            emb = HashingEmbedder()
        elif embedder_model == "local" or embedder_model.startswith("local:"):
            from .embed import LocalOnnxEmbedder
            try:
                emb = CachedEmbedder(
                    LocalOnnxEmbedder(embedder_model, cache_dir=cache_dir))
            except SemanticBackendUnavailable as e:
                print(f"faithsift: {e}", file=sys.stderr)
                print("faithsift: 已退回词法检索(--retriever lexical)。",
                      file=sys.stderr)
        else:
            from .embed import OpenAIEmbedder
            emb = CachedEmbedder(OpenAIEmbedder(model=embedder_model))
        if emb is not None:
            dense = _DenseIndex(list(chunks), emb)
    return Retriever(chunks, backend=backend, dense=dense, **kwargs)


__all__ = ["BM25Index", "Retriever", "build_retriever", "numeric_hit"]
