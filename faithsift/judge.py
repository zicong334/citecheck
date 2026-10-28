# -*- coding: utf-8 -*-
"""判定引擎:LLM 判官(DeepSeek)+ 零 API 的启发式兜底。

两者都返回 :class:`ClaimVerdict`,`verdict` 取值
``{supported, unsupported, contradicted}``,并附上它依据的证据片段编号。
``multi_hop`` 标记"需要不止一段证据"的判定。

判定规则(两个引擎共用,提示词与规则里都写了):

* 数值必须**严格一致** ——"证据说 200 以上、论断说 180 以上"一律判
  ``contradicted``,不允许用逻辑蕴含放行;
* 论断点名实体 A 时,讲实体 B 的段落不能充当它的证据(张冠李戴);
* ``unsupported`` 的含义是"证据沉默",而不是"说错了"——这条区分正是
  三分类存在的全部意义。
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from .evidence import Chunk, distinctive_numbers, tokenize
from .lexicon import concepts, entities
from .numbers import normalize_numerals
from .retrieval import Retriever

VERDICTS = ("supported", "unsupported", "contradicted")

JUDGE_SYSTEM = (
    "你是引用忠实性审核员。给定一条【论断】和若干带编号的【证据片段】,"
    "判断论断是否被证据支持。判定标准:\n"
    "- supported: 证据明确包含论断的核心事实(实体与关系一致;非数值事实的同义改写、"
    "跨语言翻译算支持)。若一条论断需要综合**多个**片段才能完整成立,把用到的片段编号"
    "全部列出,这属于多跳支持,仍判 supported\n"
    "- contradicted: 证据与论断核心事实冲突。**数值必须严格一致**:论断数字与证据数字"
    "不同时一律判 contradicted,即使逻辑上可蕴含(证据说\"200度以上\",论断说\"180度以上\","
    "属于篡改数值,判 contradicted);方向相反(升高/降低、高于/低于)、实体张冠李戴、"
    "单位不同(9 bar vs 15 bar)同理\n"
    "- unsupported: 证据中找不到论断主题的依据(既不支持也不冲突)。注意:证据提到相邻话题"
    "但没有论断所说的这个事实,属于 unsupported,不是 contradicted\n"
    "只依据给定证据判断,不使用你自己的世界知识。证据可能是英文而论断是中文(或相反),"
    "跨语言时按语义对照判断。\n"
    '只输出一个 JSON 对象: {"verdict": "supported|unsupported|contradicted", '
    '"evidence_idx": <最相关片段编号;多跳时输出编号数组,如 [2,5];无则 -1>, '
    '"quote": "<=40字的关键证据原文摘录", "reason": "<=30字中文理由"}'
)

_JSON_RETRY = ("上次输出不是合法 JSON。请只输出一个 JSON 对象,不要任何解释或 markdown:"
               '{"verdict":"...","evidence_idx":-1,"quote":"...","reason":"..."}')

# --------------------------------------------------------------------------
# 方向/量级极性:"above 200" 与 "below 200" 是冲突,不是改写。
# 启发式引擎用它做判定,LLM 提示词里用自然语言表述同一规则。
# --------------------------------------------------------------------------
_UP_WORDS = (
    "以上", "超过", "高于", "大于", "至少", "不少于", "多于", "上升", "增加",
    "提高", "增长", "升高", "提升", "促进", "增强",
    "越大", "越高", "越深", "越强", "更多", "更长",
    "above", "over", "more than", "at least", "higher", "exceed", "exceeds",
    "increase", "increases", "increased", "increasing", "rise", "rises",
    "rose", "grow", "grows", "grew", "gain", "gained", "promote", "promotes",
)
_DOWN_WORDS = (
    "以下", "低于", "小于", "不足", "不超过", "最多", "至多", "下降", "降低",
    "减少", "减小", "缩减", "抑制", "削弱",
    "越小", "越低", "越浅", "越弱", "更少", "更短",
    "below", "under", "less than", "at most", "lower", "decrease",
    "decreases", "decreased", "decreasing", "reduce", "reduces", "reduced",
    "drop", "drops", "dropped", "fell", "fall", "falls", "cut", "cuts",
    "inhibit", "inhibits", "suppress", "suppresses",
)
_UP_RE = re.compile("|".join(sorted((re.escape(w) for w in _UP_WORDS),
                                   key=len, reverse=True)), re.I)
_DOWN_RE = re.compile("|".join(sorted((re.escape(w) for w in _DOWN_WORDS),
                                     key=len, reverse=True)), re.I)
_NUM = re.compile(r"\d+(?:\.\d+)?")
_WINDOW_BEFORE = 12
_WINDOW_AFTER = 8
_CONCEPT_WINDOW = 12


def _polarity_at(seg: str, reverse: bool) -> int | None:
    """取 *seg* 里最后一个(或第一个)方向词:+1 升、-1 降、None 没有。"""
    hits: list[tuple[int, int]] = []
    for m in _UP_RE.finditer(seg):
        hits.append((m.start(), 1))
    for m in _DOWN_RE.finditer(seg):
        hits.append((m.start(), -1))
    if not hits:
        return None
    hits.sort()
    return hits[-1][1] if reverse else hits[0][1]


def polarities(text: str) -> dict[str, set[int]]:
    """把 *text* 里每个数字映射到挂在它身上的方向词。

    ``"降低约 60% 至 75%"`` → ``{"60": {-1}, "75": {-1}}``;
    附近没有方向词的数字不会出现在结果里(没有方向证据就不猜方向)。
    """
    out: dict[str, set[int]] = {}
    for m in _NUM.finditer(text):
        before = text[max(0, m.start() - _WINDOW_BEFORE):m.start()]
        after = text[m.end():m.end() + _WINDOW_AFTER]
        pol = _polarity_at(before, reverse=True)
        if pol is None:
            pol = _polarity_at(after, reverse=False)
        if pol is not None:
            out.setdefault(m.group(), set()).add(pol)
    return out


def _num_mentions(text: str) -> list[tuple[str, int | None, frozenset[str]]]:
    """每个数字出现处的 (数值, 方向, 附近概念) 三元组。"""
    out: list[tuple[str, int | None, frozenset[str]]] = []
    for m in _NUM.finditer(text):
        before = text[max(0, m.start() - _WINDOW_BEFORE):m.start()]
        after = text[m.end():m.end() + _WINDOW_AFTER]
        pol = _polarity_at(before, reverse=True)
        if pol is None:
            pol = _polarity_at(after, reverse=False)
        window = text[max(0, m.start() - _WINDOW_BEFORE):m.end() + _WINDOW_AFTER]
        out.append((m.group(), pol, frozenset(concepts(window))))
    return out


_SLOT_BEFORE = 16
_SLOT_AFTER = 12


def num_slots(text: str) -> list[tuple[str, frozenset[str], frozenset[str]]]:
    """每个数字出现处的 (数值, 窗口概念, 窗口实体)。

    数值先经 :func:`normalize_numerals` 归一化,使其与
    :func:`distinctive_numbers` 的口径一致(``两千``→``2000``、去千分位)。
    **不做**区分度过滤:单位错误这类用例里,证据的真值可能是个位数
    (``9 bar``),若按 ``distinctive_numbers`` 的"≥2 位"口径丢掉它,就找不到
    与论断 ``15 bar`` 竞争的取值了。是否区分度足够由调用方按场景决定。
    窗口概念/实体用来界定"这个数字讲的是哪一件事"——判断证据里的某个数值
    能否充当论断缺失值的"竞争取值"时,必须落在同一个事实槽(共享区分度概念)、
    且归属于同一实体。
    """
    norm = normalize_numerals(text)
    out: list[tuple[str, frozenset[str], frozenset[str]]] = []
    for m in _NUM.finditer(norm):
        num = m.group()
        before = norm[max(0, m.start() - _SLOT_BEFORE):m.start()]
        after = norm[m.end():m.end() + _SLOT_AFTER]
        out.append((num, frozenset(concepts(before) | concepts(after)),
                    frozenset(entities(before) | entities(after))))
    return out


def polarity_conflict(claim: str, evidence_text: str) -> tuple[str, set[int], set[int]] | None:
    """同一个数字在论断与证据里方向相反,且两处讨论的是同一个概念。

    加"同一概念"这道门槛是必要的:论断"绿地面积每增加 10%"与证据
    "表面温度降低 10 至 20 摄氏度"共享数字 10,却毫不相干,不加约束会把
    一条"证据沉默"误判成"方向冲突"。
    """
    for num, pol, cons in _num_mentions(claim):
        if pol is None:
            continue
        for enum, epol, econs in _num_mentions(evidence_text):
            if enum != num or epol is None or pol is epol:
                continue
            if cons & econs:
                return num, {pol}, {epol}
    return None


def concept_polarity(text: str) -> dict[str, set[int]]:
    """方向词 -> 它后面最近的概念,形成 ``(概念, 方向)`` 配对。

    用来抓**没有数字**的方向反转:证据说"提高茶氨酸含量",论断说
    "降低茶氨酸含量",词面上高度重合,数字检查完全失效。
    只看方向词之后 ``_CONCEPT_WINDOW`` 个字符内的概念,避免跨句乱配。
    """
    out: dict[str, set[int]] = {}
    for regex, pol in ((_UP_RE, 1), (_DOWN_RE, -1)):
        for m in regex.finditer(text):
            tail = text[m.end(): m.end() + _CONCEPT_WINDOW]
            for concept in concepts(tail):
                out.setdefault(concept, set()).add(pol)
    return out


def concept_polarity_conflict(claim: str, evidence_text: str) -> tuple[str, set[int], set[int]] | None:
    """同一概念在论断与证据里方向相反(且证据没有同时给出两个方向)。"""
    cp = concept_polarity(claim)
    if not cp:
        return None
    ep = concept_polarity(evidence_text)
    for concept, pols in cp.items():
        other = ep.get(concept)
        if other and not (pols & other):
            return concept, pols, other
    return None


# --------------------------------------------------------------------------
# 定性量级反转:有些方向不挂在数字上,也不挂在"提高/降低"这类动词上,而是
# 挂在一个**定性刻度**的两端(烘焙越"深" vs 越"浅"、颜色越"暗" vs 越"亮")。
# 典型用例:论断"Agtron 数值越大表示烘焙程度越深",源文却是"higher numbers
# indicate lighter roasts"——两侧都在讲"数值↑ → 烘焙程度",但一端是深、一端是浅。
# 数字校验、方向词配对都抓不到它(方向词"越大/higher"两侧其实同向),
# 只有把"深/浅"当作烘焙刻度的一对反义词才能识破。
# 为避免"越深/越浅"在别的话题(如"睡眠越深")误伤,强制要求两侧都处于烘焙语境。
# --------------------------------------------------------------------------
_ROAST_DARK = ("深烘", "深度烘焙", "烘焙程度越深", "越深", "darker", "dark roast")
_ROAST_LIGHT = ("浅烘", "浅度烘焙", "烘焙程度越浅", "越浅", "lighter", "light roast")


def _roast_context(text: str) -> bool:
    low = text.lower()
    return "烘焙" in text or "roast" in low or "agtron" in low


def roast_level(text: str) -> set[int]:
    """文本里出现的烘焙深浅极性:+1 深、-1 浅(可同时出现,表示两侧都提到)。"""
    low = text.lower()
    poles: set[int] = set()
    if any(w in text or w in low for w in _ROAST_DARK):
        poles.add(1)
    if any(w in text or w in low for w in _ROAST_LIGHT):
        poles.add(-1)
    return poles


def qualitative_conflict(claim: str, evidence_text: str) -> tuple[str, set[int], set[int]] | None:
    """烘焙语境下,论断与证据把同一刻度指向相反两端(深 vs 浅)。

    只在两侧都提到烘焙、且各自只指向一端、方向相反时才判冲突;
    若证据同时提到深与浅(如综述两种烘焙),交集非空 -> 不判冲突。
    """
    if not (_roast_context(claim) and _roast_context(evidence_text)):
        return None
    cl = roast_level(claim)
    el = roast_level(evidence_text)
    if cl and el and not (cl & el):
        return "烘焙程度", cl, el
    return None


def _cjk_ratio(text: str) -> float:
    """中文字符占"中日韩 vs 拉丁字母"的比例,用来判断语种。

    只在中文字符与拉丁字母之间取比例,**忽略数字、标点与空白**:否则一段
    数字/百分号密集的中文证据(如"…15 辆…30%…1990 年代…helmet(头盔)…")
    会因为阿拉伯数字拉低中文占比,被误判成"非中文",进而让一条同语言论断
    被当跨语言处理、错误地走上概念捷径。
    """
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    denom = cjk + latin
    return cjk / denom if denom else 0.0


def _is_cross_lingual(a: str, b: str) -> bool:
    """一侧以中文为主、另一侧以拉丁字母为主 -> 视为跨语言。"""
    return abs(_cjk_ratio(a) - _cjk_ratio(b)) > 0.2


@dataclass
class ClaimVerdict:
    claim: str
    verdict: str
    mode: str
    score: float = 0.0
    evidence_idx: int = -1
    evidence_doc: str = ""
    quote: str = ""
    reason: str = ""
    evidence_idxs: list[int] = field(default_factory=list)
    multi_hop: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.evidence_idxs and self.evidence_idx >= 0:
            self.evidence_idxs = [self.evidence_idx]


class BudgetStop(RuntimeError):
    pass


class Ledger:
    """只追加、带硬顶(人民币)的成本账本。"""

    def __init__(self, path: Path, cap: float | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cap = cap if cap is not None else float(
            os.environ.get("FAITHSIFT_CAP")
            or os.environ.get("CITECHECK_CAP")  # 旧名回退(CiteCheck → FaithSift)
            or "12")
        self.total = 0.0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.total += float(json.loads(line).get("cost_cny", 0))

    def charge(self, model: str, pt: int, ct: int, note: str = "") -> float:
        cost = pt / 1e6 * 2.0 + ct / 1e6 * 8.0  # DeepSeek chat list price
        rec = {"ts": round(time.time(), 2), "model": model, "prompt_tokens": pt,
               "completion_tokens": ct, "cost_cny": round(cost, 6), "note": note}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.total += cost
        if self.total > self.cap:
            raise BudgetStop(f"ledger total {self.total:.2f} > cap {self.cap:.2f}")
        return cost


def _parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    else:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            text = text[s:e + 1]
    return json.loads(text)


def _load_api_key(explicit: str | None = None) -> str | None:
    """按优先级取 DeepSeek key,绝不把密钥写进仓库文件。

    顺序:显式传入 > 进程环境变量 > ``FAITHSIFT_ENV_FILE`` 指向的文件
    (旧名 ``CITECHECK_ENV_FILE`` 仍作回退) > 仓库根的 ``.env.local``(本地私有)
    > 仓库根的 ``.env``(只有占位符)。
    """
    key = explicit or os.environ.get("DEEPSEEK_API_KEY")
    if key and key.strip() and not key.startswith("sk-your"):
        return key.strip()
    candidates: list[Path] = []
    env_file_var = (os.environ.get("FAITHSIFT_ENV_FILE")
                    or os.environ.get("CITECHECK_ENV_FILE"))  # 旧名回退
    if env_file_var:
        candidates.append(Path(env_file_var))
    root = Path(__file__).resolve().parent.parent
    candidates.extend([root / ".env.local", root / ".env"])
    for env_file in candidates:
        if not env_file.exists():
            continue
        from dotenv import dotenv_values
        value = (dotenv_values(env_file).get("DEEPSEEK_API_KEY") or "").strip()
        if value and not value.startswith("sk-your"):
            return value
    return None


class LLMJudge:
    mode = "llm"

    def __init__(self, model: str = "deepseek-chat", ledger: Ledger | None = None,
                 api_key: str | None = None, base_url: str = "https://api.deepseek.com"):
        from openai import OpenAI
        key = _load_api_key(api_key)
        if not key:
            raise RuntimeError(
                "未找到 DeepSeek 密钥。请设置环境变量 DEEPSEEK_API_KEY,"
                "或把 .env.example 复制为 .env.local 后填写;"
                "离线自检请改用 --mode heuristic")
        self.client = OpenAI(api_key=key, base_url=base_url, timeout=90, max_retries=0)
        self.model = model
        self.ledger = ledger

    def _ask(self, messages: list[dict]) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, max_tokens=300)
        u = resp.usage
        if self.ledger:
            self.ledger.charge(self.model, u.prompt_tokens, u.completion_tokens,
                               note="judge")
        return _parse_json(resp.choices[0].message.content or "")

    def judge(self, claim: str, chunks: Sequence[Chunk]) -> ClaimVerdict:
        ev = "\n\n".join(f"[{c.idx}] ({c.doc}) {c.text}" for c in chunks)
        messages = [{"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": f"【论断】\n{claim}\n\n【证据片段】\n{ev}"}]
        last: Exception | None = None
        parse_failed = False
        for attempt in range(4):
            try:
                if parse_failed:
                    messages = messages[:2] + [
                        {"role": "user", "content": _JSON_RETRY}]
                data = self._ask(messages)
                return self._to_verdict(claim, chunks, data)
            except BudgetStop:
                raise
            except (json.JSONDecodeError, ValueError) as e:
                last = e
                parse_failed = True
                time.sleep(0.5)
            except Exception as e:  # noqa: BLE001
                last = e
                s = str(e).lower()
                if "402" in s or "balance" in s or "insufficient" in s:
                    raise BudgetStop(f"API balance issue: {e}") from e
                time.sleep(min(2 ** attempt * 2, 20))
        return ClaimVerdict(claim=claim, verdict="unsupported", mode="llm_error",
                            reason=f"judge failed: {str(last)[:120]}",
                            raw={"_error": str(last)[:200],
                                 "_parse_failed": parse_failed})

    def _to_verdict(self, claim: str, chunks: Sequence[Chunk],
                    data: dict) -> ClaimVerdict:
        v = data.get("verdict")
        if v not in VERDICTS:
            v = "unsupported"
        by_idx = {c.idx: c for c in chunks}
        raw_idx = data.get("evidence_idx", -1)
        if isinstance(raw_idx, bool):
            candidates = [-1]
        elif isinstance(raw_idx, int):
            candidates = [raw_idx]
        elif isinstance(raw_idx, (list, tuple)):
            candidates = [x for x in raw_idx if isinstance(x, int) and not isinstance(x, bool)]
        else:
            candidates = [-1]
        valid = [i for i in dict.fromkeys(candidates) if i in by_idx]
        first = valid[0] if valid else -1
        doc = by_idx[first].doc if first >= 0 else ""
        return ClaimVerdict(
            claim=claim, verdict=v, mode=self.mode,
            evidence_idx=first, evidence_doc=doc,
            evidence_idxs=valid, multi_hop=len(valid) > 1,
            quote=str(data.get("quote", ""))[:80],
            reason=str(data.get("reason", ""))[:60],
            raw=data)


class HeuristicJudge:
    """零 API 的词法判官,支持多跳证据组合。

    四个信号:

    ``覆盖度``   论断的非数字词元在多大概率上被某片段覆盖。**按 IDF 加权**:
                 生僻词(``waggle_dance``)权重高,口水词(``的``/``了``)权重低,
                 因此同义改写不会像等权覆盖那样被稀释。
    ``并集``     最佳片段 + 能带来新信息量的第二片段,合并后的覆盖度。
    ``数值``     论断里的特征数字是否在证据范围内出现过。
    ``方向``     同一个数字前后的升降/高低方向词是否矛盾。

    判定顺序:方向冲突 → 实体错配 → 数值缺失 → 覆盖度。
    只有第二个片段相对最佳单片段带来至少 ``min_hop_gain`` 的增益、
    且并集覆盖度达到 ``multihop_t`` 时,才授予 ``multi_hop``——
    否则任意一段沾边的段落都会被算成支持。
    """
    mode = "heuristic"

    def __init__(self, supported_t: float = 0.55, unsupported_t: float = 0.30,
                 multihop_t: float = 0.70, max_hops: int = 2,
                 min_hop_gain: float = 0.15, widen_t: float = 0.40,
                 idf: dict[str, float] | None = None,
                 min_concept_hits: int = 3, concept_t: float = 0.70,
                 semantic_t: float = 0.75):
        self.supported_t = supported_t
        self.unsupported_t = unsupported_t
        self.multihop_t = multihop_t
        self.max_hops = max_hops
        self.min_hop_gain = min_hop_gain
        self.widen_t = widen_t
        self.min_concept_hits = min_concept_hits
        self.concept_t = concept_t
        self.semantic_t = semantic_t
        self._idf: dict[str, float] | None = None
        self._max_idf = 3.0
        self._semantic = None  # 可选向量通道(DenseIndex);接入后补一条语义证据
        if idf:
            self.attach_idf(idf)

    def attach_idf(self, idf: dict[str, float]) -> None:
        """接入检索侧的 IDF 表;不接也能跑,只是退化为等权覆盖。"""
        self._idf = dict(idf)
        self._max_idf = max(self._idf.values()) if self._idf else 3.0

    def attach_semantic(self, dense_index) -> None:
        """接入可选的向量索引(DenseIndex 协议:``cosine(query)->list[float]``)。

        接入后,判官在**词法/概念覆盖之外**多一条语义证据:论断与片段的余弦
        相似度。它专门补词法引擎的根本短板——"字面几乎不重合的同义改写"
        (四到六个周期↔4 至 6 个周期)与"凭空追加关系"(绿地→房价)在词法层
        结构相同,只有语义相似度能分开(实测前者 0.79~0.92、后者 0.40~0.57)。
        不接入(纯词法/默认)时行为完全不变。
        """
        self._semantic = dense_index

    def _semantic_sims(self, claim: str, chunks: Sequence[Chunk]) -> dict[int, float]:
        """论断与各候选片段的语义余弦;没有向量通道时返回空表。"""
        if self._semantic is None or not chunks:
            return {}
        try:
            sims = self._semantic.cosine(claim)
        except Exception:  # noqa: BLE001  向量通道出错绝不能拖垮词法判定
            return {}
        return {c.idx: float(sims[c.idx]) for c in chunks
                if 0 <= c.idx < len(sims)}

    @staticmethod
    def _nonnum_tokens(text: str) -> set[str]:
        return {t for t in tokenize(text) if not any(ch.isdigit() for ch in t)}

    def _weight(self, token: str) -> float:
        if self._idf is None:
            return 1.0
        # 语料里从未出现的词按最稀有处理:它不该帮论断拿到覆盖度
        return self._idf.get(token, self._max_idf)

    def _lexical_cov(self, tokens: set[str], chunk_tokens: set[str]) -> float:
        if not tokens:
            return 0.0
        total = sum(self._weight(t) for t in tokens)
        if total <= 0:
            return 0.0
        return sum(self._weight(t) for t in tokens & chunk_tokens) / total

    def _concept_cov(self, concept_tokens: frozenset[str],
                     chunk_tokens: set[str]) -> float:
        total = sum(self._weight(t) for t in concept_tokens)
        if total <= 0:
            return 0.0
        return sum(self._weight(t) for t in concept_tokens if t in chunk_tokens) / total

    def _has_distinctive_concept(self, claim_concepts: frozenset[str],
                                 chunk_tokens: set[str]) -> bool:
        """论断概念里是否有一个"高区分度"(语料里很稀有)且被该片段覆盖的概念。

        只有 2 个概念命中时,靠一个稀有概念(如 ``agtron``,全语料只出现一次)
        就足以确认"讲的是同一件事";而 ``coffee``/``roast`` 这类高频概念不够,
        否则"咖啡烘焙普遍采用 AI"这种泛泛论断会被顶成有据。区分度用 IDF 衡量:
        达到全表最高 IDF 的 95%(≈全语料只出现一次)视为稀有,足以锚定"同一件事"。
        """
        if not self._idf:
            return False
        thresh = 0.95 * self._max_idf
        return any(self._idf.get(c, 0.0) >= thresh
                   for c in claim_concepts if c in chunk_tokens)

    def _distinctive_concepts(self, cons: frozenset[str]) -> frozenset[str]:
        """从概念集合里挑出高区分度(语料稀有)的那些。"""
        if not self._idf:
            return cons
        thresh = 0.7 * self._max_idf
        return frozenset(c for c in cons
                         if self._idf.get(c, self._max_idf) >= thresh)

    def _competing_value_exists(self, claim: str, missing: set[str],
                                scope: Sequence[Chunk],
                                claim_ents: set[str]) -> bool:
        """证据里是否存在能与论断缺失数字"竞争同一事实槽"的取值。

        数字篡改(应判 contradicted)与相邻话题(应判 unsupported)的唯一区别:
        前者证据就**同一事实**给出了另一个取值(16↔15、59%↔49%),后者证据
        根本没讲论断所说的那件事。判据:证据里某个数字,若与缺失数字共享一个
        区分度概念(同一事实槽),且不是明确归属于**另一个**实体,则视为竞争取值。
        找不到竞争取值 -> 语料对该事实沉默 -> unsupported(相邻话题≠冲突)。

        只在论断**点名了实体**时才做这层筛查:没有实体时,"主题覆盖高 + 数字缺失"
        就是数字篡改的经典特征(咖啡 96→106、蜂蜜 190 万→210 万),沿用旧判据即可;
        否则跨语言、窗口不对称会让"同槽概念"匹配不上,把真篡改误降为沉默。
        """
        if not claim_ents:
            return True
        missing_slots = [self._distinctive_concepts(cons - ents)
                         for num, cons, ents in num_slots(claim) if num in missing]
        # 缺失数字都定位不到区分度概念时,保守沿用旧行为(判冲突),不误伤数字篡改。
        if not any(missing_slots):
            return True
        for c in scope:
            for num, cons, ents in num_slots(c.text):
                if num in missing:
                    continue
                # 实体门:证据数字若明确只讲"另一个"实体,不能反驳论断实体的事实。
                if claim_ents and ents and not (ents & claim_ents):
                    continue
                # 事实槽只比"讲的是什么"(去掉实体词元,实体已由上面的实体门把关)。
                ev_slot = self._distinctive_concepts(cons - ents)
                if any(slot & ev_slot for slot in missing_slots):
                    return True
        return False

    def _coverage(self, tokens: set[str], chunk_tokens: set[str],
                  claim_concepts: frozenset[str] = frozenset(),
                  claim_nums: set[str] | None = None,
                  chunk_nums: set[str] | None = None,
                  cross: bool = True, two_concept_ok: bool = True) -> float:
        """综合覆盖度 = max(词法加权覆盖, 概念覆盖)。

        跨语言场景下,中文单字/双字组永远不可能出现在英文块里,只用词法覆盖
        会让"中文论断 vs 英文证据"恒低于阈值。因此额外计算**概念覆盖**:
        两侧共享的概念词元(含实体名、数量级)占论断概念的比例。

        关键取舍:**同语言单块不走概念捷径**。同语言时词法覆盖已经能反映
        "论断里有没有证据之外的主题词";若再用共享概念把分数顶上去,
        "透水铺装寿命比沥青长一倍""绿地面积每增加 10%,房价上涨 3%"这种
        只沾主题词、实则语料沉默的论断就会被放过。注意"四到六个睡眠周期、
        每个九十分钟"这类**字面改写但数值忠实**的同义句,与"绿地+房价"这种
        **凭空追加关系**的论断,在词法层结构完全一样(都是概念全中+数字命中),
        无法区分——重度同义改写只能交给**向量通道**(见 :mod:`faithsift.embed`)。
        多跳的并集另见 ``_select_hops``:跨段落论断天然缺字面重合。
        """
        lex = self._lexical_cov(tokens, chunk_tokens)
        if not cross:
            return lex
        n = len(claim_concepts)
        if n < 2:
            return lex
        sem = self._concept_cov(claim_concepts, chunk_tokens)
        if sem < self.concept_t:
            return lex
        if n >= self.min_concept_hits:
            return max(lex, sem)
        # 恰好 2 个概念的宽松捷径只留给跨语言**单块**匹配(如 agtron+roast 认定
        # "讲的就是同一件事"):要么数字全部命中,要么含一个高区分度概念。
        # 多跳并集不走这条(two_concept_ok=False)——两段证据各沾一个主题词就把
        # "共享单车维护成本由押金覆盖"这类相邻话题凑成支持,太弱;多跳要么≥3 概念,
        # 要么靠词法并集。同语言单块已被上面的 `not cross` 挡掉。
        if two_concept_ok and sem >= 0.999 and (
                (claim_nums and chunk_nums is not None and claim_nums <= chunk_nums)
                or self._has_distinctive_concept(claim_concepts, chunk_tokens)):
            return max(lex, sem)
        return lex

    def _select_hops(self, claim: str, nt: set[str], chunks: Sequence[Chunk],
                     claim_concepts: frozenset[str] = frozenset(),
                     claim_nums: set[str] | None = None) \
            -> tuple[list[Chunk], float, dict[int, float]]:
        cov = {c.idx: self._coverage(nt, c.tokens, claim_concepts, claim_nums,
                                     distinctive_numbers(c.text),
                                     _is_cross_lingual(claim, c.text))
               for c in chunks}
        ranked = sorted(chunks, key=lambda c: -cov[c.idx])
        best = ranked[0]
        hops = [best]
        union = set(best.tokens)
        total = cov[best.idx]
        for c in ranked[1:]:
            if len(hops) >= self.max_hops:
                break
            merged = union | c.tokens
            # 多跳并集始终允许概念辅助:跨段落论断把两段事实用"而/以及"等连接词
            # 拼起来,字面重合天然偏低,只有概念并集能反映"两段合起来是否覆盖了论断"。
            # 但并集不走"恰好 2 概念"的宽松捷径(two_concept_ok=False):那太弱,
            # 会让两段各沾一个主题词的相邻话题被凑成多跳支持。
            grown = self._coverage(nt, merged, claim_concepts, claim_nums,
                                   None, True, two_concept_ok=False)
            if grown - total >= self.min_hop_gain:
                hops.append(c)
                union = merged
                total = grown
        return hops, total, cov

    def judge(self, claim: str, chunks: Sequence[Chunk]) -> ClaimVerdict:
        if not chunks:
            return ClaimVerdict(claim=claim, verdict="unsupported", mode=self.mode,
                                reason="无候选证据")

        # 证据分区:
        #   primary  —— 提到论断所点实体,或没提任何实体的片段(可以直接当证据)
        #   others   —— 只提到**别的**实体的片段(不能直接当证据,但可能是多跳的另一半)
        # 论断没点实体时,全部算 primary。
        claim_ents = entities(claim)
        ents_of = {c.idx: _chunk_entities(c.text) for c in chunks}
        nt = self._nonnum_tokens(claim)
        claim_concepts = frozenset(concepts(claim))
        claim_nums = distinctive_numbers(claim)
        # 可选语义通道:论断与各候选片段的余弦。纯词法时为 {} -> sem_best=0,
        # 下面所有 `sem_best >= semantic_t` 的判断自动失效,行为与 v0.2 一致。
        sem_sims = self._semantic_sims(claim, chunks)
        sem_best = max(sem_sims.values(), default=0.0)
        sem_chunk = max(sem_sims, key=sem_sims.get) if sem_sims else -1

        primary, others = list(chunks), []
        forced_mismatch: tuple[set[str], set[str]] | None = None
        if claim_ents:
            primary = [c for c in chunks
                       if not ents_of[c.idx] or (ents_of[c.idx] & claim_ents)]
            others = [c for c in chunks if c not in primary]
            if not primary:
                # 候选片段全都在讲别的实体 -> 这就是张冠李戴,不要回退成"都能用"。
                # 在题判定兼顾语义:跨语言张冠李戴(巴西↔中国蜂蜜产量)词法覆盖
                # 极低,但语义高度在题,只有向量通道能认出"讲的是同一件事、实体错了"。
                ontopic = [c for c in chunks
                           if sem_sims.get(c.idx, 0.0) >= self.semantic_t
                           or self._coverage(nt, c.tokens, claim_concepts,
                                             claim_nums,
                                             distinctive_numbers(c.text),
                                             _is_cross_lingual(claim, c.text))
                           >= self.unsupported_t]
                if ontopic:
                    named: set[str] = set()
                    for c in ontopic:
                        named |= ents_of[c.idx]
                    forced_mismatch = (claim_ents, named)
                primary = list(chunks)

        hops, union_cov, cov = self._select_hops(claim, nt, primary,
                                                 claim_concepts, claim_nums)
        best = hops[0]
        single_cov = cov[best.idx]

        # 数字"存在性"只认在题的片段:一个无关段落里恰好出现同样的数字,
        # 不能算作这条论断的证据(否则"20 亿"会被"20 天"顶替)。
        scope = list(hops)
        for c in primary:
            if c not in scope and cov.get(c.idx, 0.0) >= self.widen_t:
                scope.append(c)
        present: set[str] = set()
        for c in scope:
            present |= distinctive_numbers(c.text)
        missing = claim_nums - present

        # 方向冲突只在主题确实对得上的片段上判定,避免无关片段里同款数字误伤
        #("绿地面积增加 10%"不该被"降温 10 至 25 度"判成方向相反)。
        conflict = None
        for c in hops:
            if cov.get(c.idx, 0.0) < max(self.supported_t, self.widen_t):
                continue
            conflict = polarity_conflict(claim, c.text)
            if not conflict:
                conflict = concept_polarity_conflict(claim, c.text)
            if not conflict:
                conflict = qualitative_conflict(claim, c.text)
            if conflict:
                break

        # 数字全部落在"别的实体"的片段里 -> 张冠李戴;
        # 只缺一部分 -> 合法的跨实体多跳,把那一段补进来。
        mismatch: tuple[set[str], set[str]] | None = forced_mismatch
        if missing and others and mismatch is None:
            suppliers = [c for c in others
                         if self._coverage(nt, c.tokens, claim_concepts)
                         >= self.unsupported_t]
            if suppliers:
                covered: set[str] = set()
                for c in suppliers:
                    covered |= distinctive_numbers(c.text)
                if missing <= covered:
                    if len(missing) == len(claim_nums):
                        named: set[str] = set()
                        for c in suppliers:
                            if missing & distinctive_numbers(c.text):
                                named |= ents_of[c.idx]
                        mismatch = (claim_ents, named or set().union(
                            *[ents_of[c.idx] for c in suppliers]))
                    else:
                        extra = [c for c in suppliers
                                 if missing & distinctive_numbers(c.text)][:1]
                        hops = hops + extra
                        union = set()
                        for c in hops:
                            union |= c.tokens
                        union_cov = self._coverage(nt, union, claim_concepts)
                        present = set()
                        for c in hops:
                            present |= distinctive_numbers(c.text)
                        missing = claim_nums - present

        multi = len(hops) > 1
        idxs = [c.idx for c in hops]

        def verdict(v: str, reason: str, *, score: float | None = None) -> ClaimVerdict:
            return ClaimVerdict(
                claim=claim, verdict=v, mode=self.mode,
                score=round(single_cov if score is None else score, 3),
                evidence_idx=best.idx, evidence_doc=best.doc,
                evidence_idxs=idxs, multi_hop=multi, reason=reason)

        if conflict:
            key, cp, ep = conflict
            kind = "数值" if key[:1].isdigit() else "概念"
            return verdict("contradicted",
                           f"{kind}{key}方向相反(论断{_pol_zh(cp)}/证据{_pol_zh(ep)})")
        if mismatch:
            want, got = mismatch
            return verdict("contradicted",
                           f"实体错配: 论断指向 {','.join(sorted(want))} "
                           f"但证据是 {','.join(sorted(got)[:2])}")
        if missing:
            # 在题判定只用词法/概念覆盖,不用语义余弦:语义相似度量的是"话题接近",
            # 不是"事实存在"——"中国茶叶 2025 年研发预算 4500 万美元"与"中国茶叶产量"
            # 话题高度相似,但预算这件事语料从没讲过,应当沉默(unsupported)而非冲突。
            if single_cov >= self.supported_t or union_cov >= self.multihop_t:
                shown = ",".join(sorted(missing)[:3])
                if self._competing_value_exists(claim, missing, scope, claim_ents):
                    return verdict("contradicted", f"主题匹配但数值无证据: {shown}")
                return verdict("unsupported",
                               f"相邻话题, 语料未就同一事实给出取值: {shown}")
            return verdict("unsupported", f"证据主题覆盖仅{single_cov:.0%}, 无依据")
        if single_cov >= self.supported_t:
            return verdict("supported", f"主题覆盖{single_cov:.0%}且数值一致")
        if union_cov >= self.multihop_t:
            return verdict("supported",
                           f"多跳支持({len(hops)}段证据, 覆盖{union_cov:.0%})",
                           score=union_cov)
        # 向量通道兜底:字面/概念都不够,但语义高度一致 -> 判支持(典型是重度同义改写)。
        # 阈值 0.75 由本语料实测标定:落到这一支(词法覆盖已不足)的论断里,忠实的
        # 重度改写余弦≈0.86,而相邻话题/凭空追加事实≈0.69~0.72,0.75 正好落在间隔里。
        # 注意语义相似 ≠ 蕴含,故只用于"支持"兜底,且排在所有冲突判定之后。
        if sem_best >= self.semantic_t:
            ev = next((c for c in chunks if c.idx == sem_chunk), best)
            return ClaimVerdict(
                claim=claim, verdict="supported", mode=self.mode,
                score=round(sem_best, 3), evidence_idx=ev.idx, evidence_doc=ev.doc,
                evidence_idxs=[ev.idx], multi_hop=False,
                reason=f"语义相似{sem_best:.0%}(向量通道), 字面改写但事实一致")
        return verdict("unsupported", f"证据主题覆盖仅{single_cov:.0%}, 无依据")


def _pol_zh(pols: set[int]) -> str:
    marks = {1: "升", -1: "降"}
    return "".join(sorted(marks.get(p, "?") for p in pols))


@lru_cache(maxsize=4096)
def _chunk_entities(text: str) -> frozenset[str]:
    """片段文本里的实体集合(带缓存:同一片段会被多条论断反复用到)。"""
    return frozenset(entities(text))


def judge_claims(claims: list[str], chunks: Sequence[Chunk], judge,
                 top_k: int = 3,
                 retriever: Retriever | None = None) -> list[ClaimVerdict]:
    ret = retriever or Retriever(chunks)
    if hasattr(judge, "attach_idf"):  # 启发式判官需要检索侧的 IDF 表
        judge.attach_idf(ret.bm25.idf)
    # 若检索器带了向量索引(hybrid/dense + 嵌入后端),把它交给判官补语义证据。
    if hasattr(judge, "attach_semantic") and getattr(ret, "dense", None) is not None:
        judge.attach_semantic(ret.dense)
    return [judge.judge(claim, [c for c, _ in ret.search(claim, k=top_k)])
            for claim in claims]


__all__ = [
    "VERDICTS", "BudgetStop", "ClaimVerdict", "HeuristicJudge", "Ledger",
    "LLMJudge", "judge_claims", "polarities", "polarity_conflict",
]
