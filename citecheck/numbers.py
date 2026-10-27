# -*- coding: utf-8 -*-
"""数词归一化:让中文数词能和阿拉伯数字直接比对。

中文回答经常把数字写成汉字("两千粒卵""四到六个周期""一成含水率"),
而检索回来的源文档用的是阿拉伯数字("2,000 eggs""4 to 6 cycles")。
纯词法匹配会同时丢掉检索信号和数值一致性校验,一条被篡改的中文数词
就这样溜过去了。

本模块在分词之前把这些表面形式改写成阿拉伯数字,于是:

* ``两千`` → ``2000``(去掉千分位后与 ``2,000`` 对齐)
* ``四到六`` → ``4到6``
* ``九十`` → ``90``、``三百五十五`` → ``355``、``十五`` → ``15``
* ``一成`` → ``10%``、``百分之一`` → ``1%``
* ``10 万`` → ``100000``、``1.9 million`` → ``1900000``
* ``2,000`` → ``2000``(去掉千分位)

只改写"归一化视图",片段原文与证据摘录保持原样。归一化是幂等的,
并且对论断与源文档施加同一套规则,不存在单边收益或损失。
"""
from __future__ import annotations

import re

_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
_CN_BIG = {"万": 10 ** 4, "亿": 10 ** 8}
_CN_RUN = re.compile(r"[零〇一二三四五六七八九十百千万亿两]+")
_CN_CHENG = re.compile(r"([零〇一二两三四五六七八九])成")
_CN_PERCENT = re.compile(r"百分之([零〇一二三四五六七八九十百两]+)")
_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_CN_MAGNITUDE = {"千": 10 ** 3, "万": 10 ** 4, "亿": 10 ** 8}
_CN_MAG_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(亿|万|千)")
_EN_MAGNITUDE = {"thousand": 10 ** 3, "million": 10 ** 6,
                 "billion": 10 ** 9, "trillion": 10 ** 12}
_EN_MAG_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(trillion|billion|million|thousand)",
                        re.I)


def apply_magnitude(text: str) -> str:
    """把量级词换算成完整数字:``10 万`` -> ``100000``、``1.9 million`` -> ``1900000``。

    中英各自习惯不同(中文爱用"万/亿",英文爱用 million/billion)。不换算的话,
    "190 万吨"与"1.9 million tonnes"会被当成两个不同的数,一条其实有据的论断
    会被误判为数值冲突。
    """
    def _cn(m: re.Match) -> str:
        return str(int(round(float(m.group(1)) * _CN_MAGNITUDE[m.group(2)])))

    def _en(m: re.Match) -> str:
        return str(int(round(float(m.group(1)) * _EN_MAGNITUDE[m.group(2).lower()])))

    return _EN_MAG_RE.sub(_en, _CN_MAG_RE.sub(_cn, text))


def cn_to_int(text: str) -> int | None:
    """解析一串中文数词:``'三百五十五' -> 355``。"""
    if not text:
        return None
    total = section = number = 0
    for ch in text:
        if ch in _CN_DIGITS:
            number = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            unit = _CN_UNITS[ch]
            section += (number or 1) * unit
            number = 0
        elif ch in _CN_BIG:
            section += number
            total += (section or 1) * _CN_BIG[ch]
            section = number = 0
        else:
            return None
    return total + section + number


def strip_separators(text: str) -> str:
    """去掉千分位:``'1,900,000' -> '1900000'``(反复替换直到稳定)。"""
    while True:
        new = _COMMA.sub("", text)
        if new == text:
            return new
        text = new


def normalize_numerals(text: str) -> str:
    """把中文数词与千分位改写成标准阿拉伯数字。"""
    if not text:
        return text

    def _cheng(m: re.Match) -> str:
        return f"{_CN_DIGITS[m.group(1)] * 10}%"

    def _percent(m: re.Match) -> str:
        value = cn_to_int(m.group(1))
        return f"{value}%" if value is not None else m.group(0)

    def _run(m: re.Match) -> str:
        s = m.group(0)
        # 孤立的单位字("数百"的"百"、"数十"的"十")是约数修饰,不是具体数值
        if len(s) == 1 and s in _CN_UNITS | _CN_BIG:
            return s
        value = cn_to_int(s)
        return str(value) if value is not None else s

    text = _CN_CHENG.sub(_cheng, text)
    text = _CN_PERCENT.sub(_percent, text)
    text = _CN_RUN.sub(_run, text)
    text = strip_separators(text)
    return apply_magnitude(text)


__all__ = ["apply_magnitude", "cn_to_int", "normalize_numerals",
           "strip_separators"]
