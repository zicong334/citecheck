# -*- coding: utf-8 -*-
"""多跳证据链与实体错配的离线测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from citecheck.evidence import discover_sources, load_sources
from citecheck.judge import (ClaimVerdict, HeuristicJudge, concept_polarity,
                             polarity_conflict)
from citecheck.retrieval import Retriever
from citecheck.report import summarize, to_html, to_markdown

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "gold" / "sources"


def _judge_with_idf():
    chunks = load_sources(discover_sources(SOURCES))
    ret = Retriever(chunks)
    judge = HeuristicJudge()
    judge.attach_idf(ret.bm25.idf)
    return chunks, ret, judge


def _verdict(claim):
    chunks, ret, judge = _judge_with_idf()
    return judge.judge(claim, [c for c, _ in ret.search(claim, k=3)])


def test_multihop_needs_two_paragraphs():
    v = _verdict("茶树喜欢 15 至 25 摄氏度的年均气温,而制作绿茶时杀青温度"
                 "要升到 200 摄氏度以上。")
    assert v.verdict == "supported", v.reason
    assert v.multi_hop is True
    assert len(v.evidence_idxs) >= 2
    assert len(v.evidence_doc) > 0


def test_multihop_across_documents():
    v = _verdict("阿姆斯特丹约有 38% 的出行靠自行车,哥本哈根通勤者里骑车"
                 "上班上学的约占 49%。")
    assert v.verdict == "supported", v.reason
    assert v.multi_hop is True


def test_single_hop_is_not_marked_multihop():
    v = _verdict("浓缩咖啡通常在 9 bar 压力下萃取,水温保持在 90 至 96 摄氏度。")
    assert v.verdict == "supported", v.reason
    assert v.multi_hop is False


def test_entity_swap_is_contradicted():
    """把哥本哈根的 2005 年投资安到阿姆斯特丹头上 -> 张冠李戴。"""
    v = _verdict("阿姆斯特丹自 2005 年起累计投资超过 20 亿丹麦克朗建设自行车"
                 "基础设施。")
    assert v.verdict == "contradicted", v.reason
    assert "实体错配" in v.reason


def test_unknown_entity_is_not_supported():
    v = _verdict("挪威的自行车盗窃率约为每年每千辆 15 辆。")
    assert v.verdict == "contradicted", v.reason


def test_direction_with_number_conflict():
    v = _verdict("高反射率涂层的浅色路面会使表面温度上升 10 至 20 摄氏度。")
    assert v.verdict == "contradicted", v.reason


def test_direction_without_number_conflict():
    """没有数字的方向反转,靠『方向词-概念』配对发现。"""
    v = _verdict("遮阴栽培会降低茶氨酸含量。")
    assert v.verdict == "contradicted", v.reason


def test_polarity_helpers():
    assert polarity_conflict("温度上升 10 度", "温度降低 10 度") is not None
    # 同一个数字但概念不同,不该判成方向冲突
    assert polarity_conflict("绿地面积每增加 10%", "表面温度降低 10 至 20 摄氏度") is None
    cp = concept_polarity("抑制褪黑素分泌")
    assert cp.get("melatonin") == {-1}


def test_evidence_idxs_default_from_single_index():
    v = ClaimVerdict(claim="x", verdict="supported", mode="heuristic",
                     evidence_idx=4)
    assert v.evidence_idxs == [4]
    assert v.multi_hop is False


def test_report_marks_multihop():
    v = _verdict("茶树喜欢 15 至 25 摄氏度的年均气温,而制作绿茶时杀青温度"
                 "要升到 200 摄氏度以上。")
    vd = [v.__dict__]
    summary = summarize(vd)
    assert summary["multi_hop"] == 1
    meta = {"answer": "a.md", "sources": "s/", "mode": "heuristic",
            "model": "-", "retriever": "lexical"}
    md = to_markdown(vd, summary, meta)
    html = to_html(vd, summary, meta)
    assert "多跳" in md
    assert "多跳" in html
    assert "#" in md  # 证据定位 doc#idx


def test_unsupported_is_silence_not_conflict():
    v = _verdict("咖啡烘焙企业普遍采用人工智能来控制烘焙曲线。")
    assert v.verdict == "unsupported", v.reason
