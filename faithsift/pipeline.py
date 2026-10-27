# -*- coding: utf-8 -*-
"""上层流水线:核查一份回答,以及用金标集给判官打分。"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .claims import split_claims
from .evidence import Chunk, discover_sources, load_sources
from .judge import HeuristicJudge, judge_claims
from .retrieval import build_retriever
from .report import summarize, write_reports

LABELS = ("supported", "unsupported", "contradicted")


def _make_judge(mode: str, model: str, ledger_path: Path | None):
    if mode == "llm":
        from .judge import Ledger, LLMJudge
        ledger = Ledger(ledger_path or Path("results/cost_ledger.jsonl"))
        return LLMJudge(model=model, ledger=ledger)
    if mode == "heuristic":
        return HeuristicJudge()
    raise ValueError(f"unknown mode {mode!r}")


def _attach_idf(judge, ret) -> None:
    """把检索侧的 IDF 表交给需要它的判官(启发式覆盖度加权用)。"""
    if hasattr(judge, "attach_idf"):
        judge.attach_idf(ret.bm25.idf)


def _load(answers: Path, sources: Path, retriever: str, embedder: str | None):
    chunks = load_sources(discover_sources(Path(sources)))
    ret = build_retriever(chunks, backend=retriever, embedder_model=embedder)
    return split_claims(Path(answers).read_text(encoding="utf-8")), chunks, ret


def _run(claims: list[str], chunks: list[Chunk], judge, ret, mode: str,
         top_k: int, meta: dict, out_prefix: Path | None) -> dict:
    _attach_idf(judge, ret)
    verdicts = judge_claims(claims, chunks, judge, top_k=top_k, retriever=ret)
    vd = [asdict(v) for v in verdicts]
    meta = {**meta, "mode": mode, "n_chunks": len(chunks)}
    summary = write_reports(vd, meta, out_prefix) if out_prefix else summarize(vd)
    return {"meta": meta, "summary": summary, "verdicts": vd}


def check_answer(answer_path: Path, sources_path: Path, mode: str = "heuristic",
                 model: str = "deepseek-chat", top_k: int = 3,
                 out_prefix: Path | None = None,
                 ledger_path: Path | None = None,
                 retriever: str = "lexical",
                 embedder: str | None = None) -> dict:
    """把一份回答拆成论断,并逐条对照证据源判定。

    ``mode``: ``heuristic`` (offline), ``llm`` (DeepSeek judge) or ``both``.
    ``retriever``: ``lexical`` / ``hybrid`` / ``dense``.
    ``embedder``: ``hashing`` or an OpenAI-compatible embedding model name;
    only consulted when ``retriever`` is ``hybrid``/``dense``.
    """
    claims, chunks, ret = _load(answer_path, sources_path, retriever, embedder)
    base_meta: dict[str, Any] = {
        "answer": str(answer_path), "sources": str(sources_path),
        "retriever": ret.backend,
        "embedder": embedder or "-",
        "n_claims": len(claims),
    }

    if mode == "both":
        h_judge = _make_judge("heuristic", model, ledger_path)
        h = _run(claims, chunks, h_judge, ret, "heuristic", top_k,
                 {**base_meta, "model": "-"}, None if out_prefix is None else
                 Path(str(out_prefix) + ".heuristic"))
        l_judge = _make_judge("llm", model, ledger_path)
        l = _run(claims, chunks, l_judge, ret, "llm", top_k,
                 {**base_meta, "model": model}, out_prefix)
        return {"heuristic": h, "llm": l}

    judge = _make_judge(mode, model, ledger_path)
    meta = {**base_meta, "model": model if mode == "llm" else "-"}
    return _run(claims, chunks, judge, ret, mode, top_k, meta, out_prefix)


def _per_class(cm: dict[str, dict[str, int]]) -> dict[str, dict]:
    out = {}
    for l in LABELS:
        tp = cm[l][l]
        fn = sum(cm[l].values()) - tp
        fp = sum(cm[p][l] for p in LABELS if p != l)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[l] = {"precision": round(prec, 4), "recall": round(rec, 4),
                  "f1": round(f1, 4), "n": tp + fn}
    return out


def validate_against_gold(gold_path: Path, sources_path: Path,
                          mode: str = "heuristic", model: str = "deepseek-chat",
                          top_k: int = 3, limit: int | None = None,
                          ledger_path: Path | None = None,
                          retriever: str = "lexical",
                          embedder: str | None = None,
                          progress: bool = True) -> dict:
    """用带标注的金标集给判官打分,返回混淆矩阵。

    Works for any JSONL with ``{id, claim, label}``; optional fields
    ``category``, ``hops`` and ``evidence_docs`` (present in the hard set) add
    per-category accuracy and multi-hop detection stats.
    """
    gold = [json.loads(x) for x in
            Path(gold_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    if limit:
        gold = gold[:limit]
    chunks = load_sources(discover_sources(Path(sources_path)))
    ret = build_retriever(chunks, backend=retriever, embedder_model=embedder)
    judge = _make_judge(mode, model, ledger_path)
    _attach_idf(judge, ret)

    cm = {t: {p: 0 for p in LABELS} for t in LABELS}
    details = []
    cat_hits: dict[str, list[int]] = {}
    hop_expected = hop_hit = 0
    for i, rec in enumerate(gold):
        v = judge.judge(rec["claim"], [c for c, _ in ret.search(rec["claim"], k=top_k)])
        pred = v.verdict if v.verdict in LABELS else "unsupported"
        label = rec["label"]
        cm[label][pred] += 1
        ok = int(pred == label)
        cat = rec.get("category") or "uncategorised"
        cat_hits.setdefault(cat, []).append(ok)
        if rec.get("hops", 1) and int(rec.get("hops") or 1) > 1:
            hop_expected += 1
            hop_hit += int(bool(v.multi_hop))
        details.append({"id": rec["id"], "label": label, "pred": pred,
                        "category": cat, "score": v.score,
                        "multi_hop": v.multi_hop,
                        "evidence_idxs": v.evidence_idxs,
                        "reason": v.reason, "claim": rec["claim"][:80]})
        if progress and (i + 1) % 50 == 0:
            print(f"[gold] {i + 1}/{len(gold)}")

    n = len(gold)
    correct = sum(cm[l][l] for l in LABELS)
    return {
        "mode": mode, "model": model if mode == "llm" else "-",
        "retriever": ret.backend, "embedder": embedder or "-",
        "gold": str(gold_path),
        "n": n, "accuracy": round(correct / n, 4) if n else 0.0,
        "confusion": cm, "per_class": _per_class(cm),
        "by_category": {k: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)}
                        for k, v in sorted(cat_hits.items())},
        "multi_hop": {"expected": hop_expected, "detected": hop_hit,
                      "recall": round(hop_hit / hop_expected, 4)
                      if hop_expected else None},
        "details": details,
    }


def gate_failures(summary: dict, fail_under: float | None = None,
                  fail_on_contradicted: bool = False) -> list[str]:
    """收集一次核查的 CI 门禁违规项(返回空列表表示通过)。"""
    problems: list[str] = []
    if fail_under is not None and summary.get("faithfulness", 0.0) < fail_under:
        problems.append(
            f"faithfulness {summary.get('faithfulness', 0):.1%} < {fail_under:.1%}")
    if fail_on_contradicted and summary.get("contradicted", 0):
        problems.append(f"{summary['contradicted']} contradicted claim(s)")
    return problems


__all__ = ["check_answer", "validate_against_gold", "gate_failures", "LABELS"]
