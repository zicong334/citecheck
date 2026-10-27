# -*- coding: utf-8 -*-
"""LLM 判官的解析与重试逻辑(全部用桩,不联网、不花钱)。"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faithsift.evidence import Chunk
from faithsift.judge import BudgetStop, LLMJudge


def _judge(monkeypatch, responses):
    """构造一个不联网的判官:按顺序返回给定的 JSON 响应。"""
    monkeypatch.setattr("faithsift.judge._load_api_key", lambda explicit=None: "sk-test")
    j = LLMJudge()
    calls = []

    def fake_ask(messages):
        calls.append(messages)
        item = responses[min(len(calls) - 1, len(responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    j._ask = fake_ask
    j.calls = calls
    return j


CHUNKS = [Chunk(doc="a.md", idx=2, text="证据一"),
          Chunk(doc="b.md", idx=5, text="证据二"),
          Chunk(doc="c.md", idx=7, text="证据三")]


def test_single_evidence_index(monkeypatch):
    j = _judge(monkeypatch, [{"verdict": "supported", "evidence_idx": 5,
                              "quote": "证据二", "reason": "一致"}])
    v = j.judge("某论断", CHUNKS)
    assert v.verdict == "supported"
    assert v.evidence_idx == 5
    assert v.evidence_idxs == [5]
    assert v.evidence_doc == "b.md"
    assert v.multi_hop is False


def test_multihop_index_array(monkeypatch):
    """多跳时判官返回编号数组,必须按全局 idx 映射解析(而不是列表下标)。"""
    j = _judge(monkeypatch, [{"verdict": "supported",
                              "evidence_idx": [2, 7],
                              "quote": "综合两段", "reason": "多跳"}])
    v = j.judge("需要两段证据的论断", CHUNKS)
    assert v.evidence_idxs == [2, 7]
    assert v.multi_hop is True
    assert v.evidence_idx == 2
    assert v.evidence_doc == "a.md"


def test_bogus_index_is_dropped(monkeypatch):
    j = _judge(monkeypatch, [{"verdict": "unsupported",
                              "evidence_idx": [99, "x", 5],
                              "quote": "", "reason": "无依据"}])
    v = j.judge("某论断", CHUNKS)
    assert v.evidence_idxs == [5]
    assert v.verdict == "unsupported"


def test_invalid_verdict_defaults_to_unsupported(monkeypatch):
    j = _judge(monkeypatch, [{"verdict": "maybe", "evidence_idx": -1}])
    assert j.judge("某论断", CHUNKS).verdict == "unsupported"


def test_parse_failure_is_retried(monkeypatch):
    j = _judge(monkeypatch, [json.JSONDecodeError("bad", "x", 0),
                             {"verdict": "contradicted", "evidence_idx": 5,
                              "reason": "数值冲突"}])
    v = j.judge("某论断", CHUNKS)
    assert v.verdict == "contradicted"
    assert len(j.calls) == 2
    assert "JSON" in j.calls[1][-1]["content"], "第二次调用应带上 JSON 修复指令"


def test_persistent_parse_failure_surfaces_error(monkeypatch):
    j = _judge(monkeypatch, [json.JSONDecodeError("bad", "x", 0)] * 4)
    v = j.judge("某论断", CHUNKS)
    assert v.mode == "llm_error"
    assert v.raw["_parse_failed"] is True


def test_budget_stop_propagates(monkeypatch):
    j = _judge(monkeypatch, [BudgetStop("超预算")])
    with pytest.raises(BudgetStop):
        j.judge("某论断", CHUNKS)


def test_prompt_allows_multi_hop_and_forbids_entailment():
    from faithsift.judge import JUDGE_SYSTEM
    assert "多个" in JUDGE_SYSTEM and "多跳" in JUDGE_SYSTEM
    assert "严格一致" in JUDGE_SYSTEM
    assert "180" in JUDGE_SYSTEM or "200" in JUDGE_SYSTEM
