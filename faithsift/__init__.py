# -*- coding: utf-8 -*-
"""FaithSift:RAG 回答的句子级引用忠实性检测器。

公开 API::

    from faithsift import check_answer, LLMJudge, HeuristicJudge, Retriever

最小用法::

    result = check_answer("answer.md", "sources/", mode="heuristic")
    result["summary"]["faithfulness"]

v0.2 新增:BM25 + 双语概念词表的检索器(``Retriever``)、
可选向量通道(:mod:`faithsift.embed`)、多跳证据链
(``ClaimVerdict.multi_hop``)。

v0.3 新增:本地离线向量后端 ``embed.LocalOnnxEmbedder``(``--embedder local``,
fastembed + 多语 MiniLM,不含 torch,缺依赖自动退回词法);判官可选接入语义余弦
(``HeuristicJudge.attach_semantic``)补重度同义改写;困难金标扩到 150 条。
"""
from .claims import split_claims
from .evidence import (Chunk, discover_sources, distinctive_numbers,
                       load_sources, retrieve, tokenize, tokenize_seq)
from .judge import (VERDICTS, BudgetStop, ClaimVerdict, HeuristicJudge,
                    Ledger, LLMJudge, judge_claims, polarities,
                    polarity_conflict)
from .lexicon import concepts, entities
from .pipeline import check_answer, gate_failures, validate_against_gold
from .retrieval import BM25Index, Retriever, build_retriever, numeric_hit

__version__ = "0.3.0"

__all__ = [
    "split_claims", "Chunk", "discover_sources", "distinctive_numbers",
    "load_sources", "retrieve", "tokenize", "tokenize_seq", "VERDICTS",
    "BudgetStop", "ClaimVerdict", "HeuristicJudge", "Ledger", "LLMJudge",
    "judge_claims", "polarities", "polarity_conflict", "concepts", "entities",
    "check_answer", "gate_failures", "validate_against_gold", "BM25Index",
    "Retriever", "build_retriever", "numeric_hit", "__version__",
]
