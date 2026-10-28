# -*- coding: utf-8 -*-
"""v0.3 三个弱点修复的回归测试(离线、零 API)。

对应 hard_gold 里此前错分、现已修复的四类:
  * 中文重度同义改写(paraphrase_zh)—— 同语言概念完全命中时走覆盖捷径;
  * 跨语言无数字(crosslingual)—— 概念词表补齐 ``annual average``;
  * 无数字的定性方向反转(direction)—— 烘焙"深/浅"刻度反义词;
  * 相邻话题 vs 取值冲突(near_miss)—— 缺失数字要有"同一事实槽 + 同一实体"
    的竞争取值才判 contradicted,否则语料沉默判 unsupported。

每条都直接对具体论断断言判定结果(照 test_multihop.py 风格),
另加新辅助函数(num_slots / roast_level / qualitative_conflict)的单元测试。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faithsift.evidence import discover_sources, load_sources
from faithsift.judge import (HeuristicJudge, num_slots, qualitative_conflict,
                             roast_level)
from faithsift.retrieval import Retriever

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


# --------------------------------------------------------------------------
# 中文重度同义改写:字面几乎不重合,但概念全中 -> supported
# --------------------------------------------------------------------------
def test_paraphrase_underestimate_is_supported():
    """H0012:『长期睡不够…低估自己到底有多困』对『慢性睡眠不足者会低估自己的受损程度』。"""
    v = _verdict("长期睡不够的人往往低估自己到底有多困。")
    assert v.verdict == "supported", v.reason


def test_paraphrase_concept_shortcut_needs_full_concept_match():
    """同语言概念捷径要求概念**全中**:只沾主题、凭空追加事实的论断不能被判支持。"""
    # 语料讲绿地降温,但从未提房价 -> 仍应 unsupported
    v = _verdict("研究表明,城市绿地面积每增加 10%,周边房价上涨约 3%。")
    assert v.verdict == "unsupported", v.reason


# --------------------------------------------------------------------------
# 跨语言无数字:英文论断对中文源,靠概念词表召回并判定
# --------------------------------------------------------------------------
def test_crosslingual_tea_temperature_no_digit_supported():
    """H0028:英文论断(数字写作英文单词)对中文源,概念全中 -> supported。"""
    v = _verdict("Tea bushes do best where the annual average temperature stays "
                 "between fifteen and twenty-five degrees Celsius.")
    assert v.verdict == "supported", v.reason
    assert v.evidence_doc == "tea_zh.md", v.evidence_doc


# --------------------------------------------------------------------------
# 无数字的定性方向反转:烘焙"深/浅"刻度
# --------------------------------------------------------------------------
def test_direction_agtron_roast_reversal_is_contradicted():
    """H0030:论断『Agtron 数值越大烘焙越深』,源文『higher numbers indicate lighter roasts』。"""
    v = _verdict("Agtron 数值越大表示烘焙程度越深。")
    assert v.verdict == "contradicted", v.reason


def test_roast_level_poles():
    assert roast_level("烘焙程度越深") == {1}
    assert roast_level("lighter roasts") == {-1}
    assert roast_level("深烘与浅烘") == {1, -1}
    assert roast_level("与烘焙无关的句子") == set()


def test_qualitative_conflict_requires_roast_context():
    # 烘焙语境下深 vs 浅 -> 冲突
    assert qualitative_conflict("烘焙程度越深", "higher numbers indicate lighter roasts")
    # 非烘焙语境的"越深"不应触发(避免"睡眠越深"这类误伤)
    assert qualitative_conflict("睡眠越深越好", "lighter roasts") is None
    # 两侧同向 -> 不冲突
    assert qualitative_conflict("烘焙程度越深", "darker roasts") is None


# --------------------------------------------------------------------------
# 相邻话题 vs 取值冲突:缺失数字要有同槽同实体的竞争取值才判冲突
# --------------------------------------------------------------------------
def test_near_miss_helmet_legislation_is_unsupported():
    """H0042:语料的头盔立法讲的是澳大利亚 1990 年代,不是荷兰 2010 -> 沉默。"""
    v = _verdict("荷兰的自行车头盔强制立法始于 2010 年。")
    assert v.verdict == "unsupported", v.reason


def test_number_alter_still_contradicted():
    """数字篡改(同一事实、同一实体、不同取值)必须仍判 contradicted。"""
    v = _verdict("哥本哈根通勤者中骑车上班上学的比例约为 59%。")
    assert v.verdict == "contradicted", v.reason


def test_unit_single_digit_value_still_contradicted():
    """单位/取值错误:源文真值是个位数(9 bar),竞争取值检查不能把它漏掉。"""
    v = _verdict("浓缩咖啡的标准萃取压力为 15 bar。")
    assert v.verdict == "contradicted", v.reason


def test_num_slots_excludes_entities_from_fact_slot():
    """num_slots 返回窗口概念与实体;事实槽比对时实体单独由实体门处理。"""
    slots = num_slots("荷兰自行车盗窃率约 15 辆")
    assert slots, "应至少解析出一个数字槽"
    num, cons, ents = slots[0]
    assert num == "15"
    assert "netherlands" in ents
    assert "theft" in cons  # 事实类型概念


def test_num_slots_normalises_chinese_numerals():
    slots = num_slots("蜂王每天产下两千粒卵")
    assert any(n == "2000" for n, _c, _e in slots), slots
