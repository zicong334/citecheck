# -*- coding: utf-8 -*-
"""论断拆分:把一段回答切成句子级论断(中英双语)。

拆句只看标点与换行,不做语法分析;随后过滤标题、markdown 装饰与过短的
客套句,只留下值得核验的论断。
"""
from __future__ import annotations

import re

_SPLIT = re.compile(r"(?<=[。!?;！？;\n])\s*")
_MIN_LEN_CJK = 6
_MIN_LEN_LATIN = 15


def _worth(text: str) -> bool:
    """判断一个片段是否值得核验(太短的多半是标题或语气词)。"""
    stripped = text.strip()
    if not stripped:
        return False
    cjk = sum(1 for ch in stripped if "一" <= ch <= "鿿")
    if cjk >= 4:
        return len(stripped) >= _MIN_LEN_CJK
    return len(stripped) >= _MIN_LEN_LATIN and any(
        ch.isalnum() for ch in stripped)


def split_claims(answer: str) -> list[str]:
    """按句切分论断,保持原顺序,并剔除无信息量的样板句。

    会跳过 markdown 标题(标题是章节名,不是可核验的事实),
    并剥掉列表符号与加粗装饰,但保留正文内容。
    """
    out: list[str] = []
    for raw in _SPLIT.split(answer):
        text = raw.strip()
        if not text:
            continue
        if text.startswith("#"):  # 标题行不是论断
            continue
        # 去掉 markdown 装饰,但保留其中的文字
        text = re.sub(r"^[-*>\s]+", "", text)
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        if not _worth(text):
            continue
        out.append(text)
    return out
