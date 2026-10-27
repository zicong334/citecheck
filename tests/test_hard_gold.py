# -*- coding: utf-8 -*-
"""困难金标与合成金标的回归基线。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faithsift.pipeline import validate_against_gold

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "gold" / "gold.jsonl"
HARD = ROOT / "gold" / "hard_gold.jsonl"
SOURCES = ROOT / "gold" / "sources"


def _load(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()
            if x.strip()]


def test_hard_gold_exists_and_structured():
    assert HARD.exists(), "运行 python -m faithsift gold 生成困难金标"
    recs = _load(HARD)
    assert len(recs) >= 40
    labels = {r["label"] for r in recs}
    assert labels == {"supported", "unsupported", "contradicted"}
    cats = {r["category"] for r in recs}
    for must in ("crosslingual", "multi_hop", "entity_swap", "direction",
                 "number_alter", "plausible_absent"):
        assert must in cats, f"缺少类别 {must}"
    assert all(r["rationale"] for r in recs), "每条都要有标注理由"
    assert all(r["id"].startswith("H") for r in recs)


def test_hard_gold_multihop_records_are_consistent():
    for r in _load(HARD):
        if r["hops"] > 1:
            assert r["evidence_docs"], f"{r['id']} 多跳必须给出来源文档"


def test_hard_gold_numbers_are_not_in_corpus():
    """number_alter 类用例篡改后的数字,必须在语料里确实不存在。"""
    from faithsift.evidence import distinctive_numbers
    corpus = "\n".join(p.read_text(encoding="utf-8")
                       for p in sorted(SOURCES.glob("*.md")))
    corpus_nums = distinctive_numbers(corpus)
    for r in _load(HARD):
        if r["category"] != "number_alter":
            continue
        absent = distinctive_numbers(r["claim"]) - corpus_nums
        assert absent, f"{r['id']} 并没有真正篡改出语料外的数字"


def test_synthetic_gold_regression_floor():
    res = validate_against_gold(GOLD, SOURCES, mode="heuristic", progress=False)
    assert res["accuracy"] >= 0.97, res["confusion"]


def test_hard_gold_heuristic_floor():
    """困难金标是能力下限:掉下来说明语义通道退化了。"""
    res = validate_against_gold(HARD, SOURCES, mode="heuristic", progress=False)
    assert res["accuracy"] >= 0.85, res["per_class"]
    assert res["multi_hop"]["recall"] >= 0.85, res["multi_hop"]
    for cat in ("entity_swap", "multi_hop", "number_alter", "unit"):
        assert res["by_category"][cat]["accuracy"] >= 0.66, res["by_category"]


def test_validation_reports_by_category():
    res = validate_against_gold(HARD, SOURCES, mode="heuristic", limit=10,
                                progress=False)
    assert res["n"] == 10
    assert set(res["by_category"]) <= {
        r["category"] for r in _load(HARD)}
    assert "retriever" in res and "gold" in res
