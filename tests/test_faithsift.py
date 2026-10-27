# -*- coding: utf-8 -*-
"""faithsift 的基础离线单元测试(不发起任何 API 调用)。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faithsift.claims import split_claims
from faithsift.evidence import discover_sources, load_sources, retrieve, tokenize
from faithsift.judge import HeuristicJudge, _parse_json
from faithsift.pipeline import check_answer, validate_against_gold
from faithsift.report import summarize, to_markdown

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "gold" / "sources"
GOLD = ROOT / "gold" / "gold.jsonl"


def test_split_claims_zh_en():
    text = "茶树适宜年均温15至25度。年降水需1000毫米以上!\n\nCoffee roasting begins with drying. OK?"
    claims = split_claims(text)
    assert any("茶树" in c for c in claims)
    assert any("Coffee" in c for c in claims)
    assert all(len(c) >= 6 for c in claims)


def test_tokenize_mixed():
    toks = tokenize("茶多酚占干重18%的Maillard反应")
    assert "茶多" in toks and "maillard" in toks and "18" in toks


def test_retrieve_finds_right_doc():
    chunks = load_sources(discover_sources(SOURCES))
    assert chunks
    hits = retrieve("茶树适宜生长的年均温度为 15 至 25 摄氏度", chunks, k=3)
    assert hits and hits[0][0].doc == "tea_zh.md"


def test_parse_json_variants():
    assert _parse_json('{"verdict": "supported"}')["verdict"] == "supported"
    assert _parse_json('```json\n{"verdict": "contradicted"}\n```')["verdict"] == "contradicted"
    assert _parse_json('前言 {"verdict": "unsupported"} 后记')["verdict"] == "unsupported"


def test_heuristic_three_way():
    chunks = load_sources(discover_sources(SOURCES))
    j = HeuristicJudge()

    def verdict(claim):
        hits = retrieve(claim, chunks, k=3)
        return j.judge(claim, [c for c, _ in hits]).verdict

    assert verdict("茶树适宜生长的年均温度为 15 至 25 摄氏度") == "supported"
    assert verdict("茶树适宜生长的年均温度为 88 至 95 摄氏度") == "contradicted"
    assert verdict("茶树在2025年的相关研发预算约为3847万美元") == "unsupported"


def test_gold_exists_and_balanced():
    assert GOLD.exists(), "run: python -m faithsift gold"
    recs = [json.loads(x) for x in GOLD.read_text(encoding="utf-8").splitlines()]
    labels = {r["label"] for r in recs}
    assert labels == {"supported", "unsupported", "contradicted"}
    assert len(recs) >= 150


def test_validate_heuristic_accuracy_floor():
    res = validate_against_gold(GOLD, SOURCES, mode="heuristic")
    assert res["accuracy"] >= 0.90, res["per_class"]


def test_check_answer_pipeline(tmp_path):
    answer = tmp_path / "answer.md"
    answer.write_text(
        "茶树适宜生长的年均温度为 15 至 25 摄氏度。"
        "阿姆斯特丹的自行车出行占比约为 61%。"
        "咖啡烘焙在2025年的研发预算约为3000万美元。\n",
        encoding="utf-8")
    out = tmp_path / "report"
    res = check_answer(answer, SOURCES, mode="heuristic", out_prefix=out)
    assert res["summary"]["n_claims"] == 3
    assert res["summary"]["supported"] == 1
    assert out.with_suffix(".md").exists() and out.with_suffix(".html").exists()
    assert out.with_suffix(".json").exists()
    md = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "忠实性得分" in md
