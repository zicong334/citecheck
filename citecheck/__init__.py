# -*- coding: utf-8 -*-
"""CiteCheck: RAG answer citation-faithfulness checker.

Public API:
    from citecheck import check_answer, LLMJudge, HeuristicJudge
"""
from .claims import split_claims
from .evidence import Chunk, discover_sources, load_sources, retrieve, tokenize
from .judge import (VERDICTS, BudgetStop, ClaimVerdict, HeuristicJudge,
                    Ledger, LLMJudge, judge_claims)
from .pipeline import check_answer, validate_against_gold

__all__ = [
    "split_claims", "Chunk", "discover_sources", "load_sources", "retrieve",
    "tokenize", "VERDICTS", "BudgetStop", "ClaimVerdict", "HeuristicJudge",
    "Ledger", "LLMJudge", "judge_claims", "check_answer",
    "validate_against_gold",
]
