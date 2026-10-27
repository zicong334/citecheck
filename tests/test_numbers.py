# -*- coding: utf-8 -*-
"""中文数词/量级归一化的离线测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from citecheck.evidence import distinctive_numbers
from citecheck.numbers import (apply_magnitude, cn_to_int, normalize_numerals,
                               strip_separators)


def test_cn_to_int_basic():
    assert cn_to_int("十五") == 15
    assert cn_to_int("二十五") == 25
    assert cn_to_int("九十") == 90
    assert cn_to_int("两百") == 200
    assert cn_to_int("三百五十五") == 355
    assert cn_to_int("一百零五") == 105
    assert cn_to_int("两千") == 2000
    assert cn_to_int("四") == 4


def test_strip_separators():
    assert strip_separators("2,000") == "2000"
    assert strip_separators("1,900,000") == "1900000"
    assert strip_separators("没有逗号 2023") == "没有逗号 2023"


def test_normalize_cn_numerals():
    assert normalize_numerals("两千粒卵") == "2000粒卵"
    assert normalize_numerals("四到六个周期") == "4到6个周期"
    assert "90" in normalize_numerals("每个周期九十分钟")
    assert "2,000" not in normalize_numerals("2,000 eggs")


def test_normalize_magnitude():
    assert normalize_numerals("10 万朵花") == "100000朵花"
    assert normalize_numerals("190 万吨") == "1900000吨"
    assert normalize_numerals("20 亿丹麦克朗") == "2000000000丹麦克朗"
    assert normalize_numerals("1.9 million tonnes") == "1900000 tonnes"
    assert normalize_numerals("about 178 million bags") == "about 178000000 bags"


def test_approximate_markers_kept():
    """『数百』是约数修饰,不能被当成 100,否则会凭空造出一个数字。"""
    assert normalize_numerals("数百种香气物质") == "数百种香气物质"
    assert distinctive_numbers("数百种香气物质") == set()


def test_percent_idioms():
    assert "10%" in normalize_numerals("一成含水率")
    assert "1%" in normalize_numerals("百分之一")


def test_distinctive_numbers_uses_normalisation():
    nums = distinctive_numbers("烘焙结束时只剩百分之一到二")
    assert "1" not in nums  # 单数字区分度太低,不参与判定
    assert distinctive_numbers("两千粒卵") == {"2000"}
    assert distinctive_numbers("约 1.9 million tonnes") == {"1900000"}


def test_apply_magnitude_idempotent():
    once = apply_magnitude("10 万")
    assert apply_magnitude(once) == once
