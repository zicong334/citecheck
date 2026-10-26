# -*- coding: utf-8 -*-
"""Verdict engines: LLM judge (DeepSeek) + zero-API heuristic fallback.

Both return ClaimVerdict with verdict in {supported, unsupported, contradicted}.
The heuristic mode makes the tool usable offline; the LLM mode is the
accuracy reference. The gold validation measures both against labelled data.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .evidence import Chunk, retrieve, tokenize

VERDICTS = ("supported", "unsupported", "contradicted")

_NUM = re.compile(r"\d+(?:\.\d+)?")

JUDGE_SYSTEM = (
    "你是引用忠实性审核员。给定一条【论断】和若干带编号的【证据片段】,"
    "判断论断是否被证据支持。判定标准:\n"
    "- supported: 证据明确包含论断的核心事实(实体与关系一致;非数值事实的同义改写、"
    "跨语言翻译算支持)\n"
    "- contradicted: 证据与论断核心事实冲突。**数值必须严格一致**:论断数字与证据数字"
    "不同时一律判 contradicted,即使逻辑上可蕴含(证据说\"200度以上\",论断说\"180度以上\","
    "属于篡改数值,判 contradicted);方向相反、实体张冠李戴同理\n"
    "- unsupported: 证据中找不到论断主题的依据(既不支持也不冲突)\n"
    "只依据给定证据判断,不使用你自己的世界知识。证据可能是英文而论断是中文(或相反),"
    "跨语言时按语义对照判断。\n"
    '只输出一个 JSON 对象: {"verdict": "supported|unsupported|contradicted", '
    '"evidence_idx": <最相关片段编号,无则 -1>, "quote": "<=40字的关键证据原文摘录", '
    '"reason": "<=30字中文理由"}'
)


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
    raw: dict[str, Any] = field(default_factory=dict)


class BudgetStop(RuntimeError):
    pass


class Ledger:
    """Append-only cost ledger with a hard cap (CNY)."""

    def __init__(self, path: Path, cap: float | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cap = cap if cap is not None else float(
            os.environ.get("CITECHECK_CAP", "12"))
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


class LLMJudge:
    mode = "llm"

    def __init__(self, model: str = "deepseek-chat", ledger: Ledger | None = None,
                 api_key: str | None = None, base_url: str = "https://api.deepseek.com"):
        from openai import OpenAI
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            env_file = os.environ.get("CITECHECK_ENV_FILE")
            if env_file and Path(env_file).exists():
                from dotenv import dotenv_values
                key = dotenv_values(env_file).get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("no DeepSeek key (env DEEPSEEK_API_KEY or CITECHECK_ENV_FILE)")
        self.client = OpenAI(api_key=key, base_url=base_url, timeout=90, max_retries=0)
        self.model = model
        self.ledger = ledger

    def judge(self, claim: str, chunks: list[Chunk]) -> ClaimVerdict:
        ev = "\n\n".join(f"[{c.idx}] ({c.doc}) {c.text}" for c in chunks)
        user = f"【论断】\n{claim}\n\n【证据片段】\n{ev}"
        last = None
        for attempt in range(4):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": JUDGE_SYSTEM},
                              {"role": "user", "content": user}],
                    temperature=0, max_tokens=300)
                u = resp.usage
                if self.ledger:
                    self.ledger.charge(self.model, u.prompt_tokens,
                                       u.completion_tokens, note="judge")
                try:
                    data = _parse_json(resp.choices[0].message.content or "")
                except (json.JSONDecodeError, ValueError):
                    data = {"verdict": "unsupported", "evidence_idx": -1,
                            "quote": "", "reason": "judge 输出不可解析"}
                v = data.get("verdict")
                if v not in VERDICTS:
                    v = "unsupported"
                idx = data.get("evidence_idx", -1)
                doc = ""
                if isinstance(idx, int) and 0 <= idx < len(chunks) and chunks[idx].idx == idx:
                    doc = chunks[idx].doc
                return ClaimVerdict(claim=claim, verdict=v, mode=self.mode,
                                    evidence_idx=idx if isinstance(idx, int) else -1,
                                    evidence_doc=doc,
                                    quote=str(data.get("quote", ""))[:80],
                                    reason=str(data.get("reason", ""))[:60],
                                    raw=data)
            except BudgetStop:
                raise
            except Exception as e:  # noqa: BLE001
                last = e
                s = str(e).lower()
                if "402" in s or "balance" in s or "insufficient" in s:
                    raise BudgetStop(f"API balance issue: {e}") from e
                time.sleep(min(2 ** attempt * 2, 20))
        return ClaimVerdict(claim=claim, verdict="unsupported", mode="llm_error",
                            reason=f"judge failed: {str(last)[:120]}")


class HeuristicJudge:
    """Zero-API lexical judge.

    Decision logic:
      topic_score  = non-numeric token containment of claim in best chunk
                     (does the evidence even talk about this?)
      numbers_ok   = every distinctive claim number appears in retrieved chunks
      - numbers_ok and topic_score high      -> supported
      - numbers missing but topic covered    -> contradicted (same fact, wrong value)
      - topic not covered                    -> unsupported (no evidence either way)
    """
    mode = "heuristic"

    def __init__(self, supported_t: float = 0.55, unsupported_t: float = 0.30):
        self.supported_t = supported_t
        self.unsupported_t = unsupported_t

    @staticmethod
    def _nonnum_tokens(text: str) -> set[str]:
        return {t for t in tokenize(text) if not any(ch.isdigit() for ch in t)}

    def judge(self, claim: str, chunks: list[Chunk]) -> ClaimVerdict:
        if not chunks:
            return ClaimVerdict(claim=claim, verdict="unsupported", mode=self.mode,
                                reason="无候选证据")
        nt = self._nonnum_tokens(claim)
        scored = sorted(
            ((c, len(nt & c.tokens) / len(nt) if nt else 0.0) for c in chunks),
            key=lambda x: -x[1])
        best, topic = scored[0]

        claim_nums = {n for n in _NUM.findall(claim) if len(n) >= 2 or "." in n}
        chunk_nums = set()
        for c in chunks:
            chunk_nums |= set(_NUM.findall(c.text))
        missing = claim_nums - chunk_nums

        def verdict(v: str, reason: str) -> ClaimVerdict:
            return ClaimVerdict(claim=claim, verdict=v, mode=self.mode,
                                score=round(topic, 3), evidence_idx=best.idx,
                                evidence_doc=best.doc, reason=reason)

        if not missing and topic >= self.supported_t:
            return verdict("supported", f"主题覆盖{topic:.0%}且数值一致")
        if missing and topic >= self.supported_t:
            return verdict("contradicted",
                           f"主题覆盖但数值无证据: {sorted(missing)[:3]}")
        if topic >= self.supported_t:
            return verdict("supported", f"主题覆盖{topic:.0%}")
        return verdict("unsupported", f"证据主题覆盖仅{topic:.0%}, 无依据")


def judge_claims(claims: list[str], chunks: list[Chunk], judge,
                 top_k: int = 3) -> list[ClaimVerdict]:
    out = []
    for claim in claims:
        hits = retrieve(claim, chunks, k=top_k)
        picked = [c for c, _ in hits]
        out.append(judge.judge(claim, picked))
    return out
