# -*- coding: utf-8 -*-
"""High-level pipelines: check an answer file, validate judges vs gold."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .claims import split_claims
from .evidence import discover_sources, load_sources, retrieve
from .judge import ClaimVerdict, HeuristicJudge, judge_claims
from .report import summarize, write_reports


def check_answer(answer_path: Path, sources_path: Path, mode: str = "heuristic",
                 model: str = "deepseek-chat", top_k: int = 3,
                 out_prefix: Path | None = None,
                 ledger_path: Path | None = None) -> dict:
    answer = Path(answer_path).read_text(encoding="utf-8")
    claims = split_claims(answer)
    chunks = load_sources(discover_sources(Path(sources_path)))
    if mode == "llm":
        from .judge import LLMJudge, Ledger
        ledger = Ledger(ledger_path or Path("results/cost_ledger.jsonl"))
        judge = LLMJudge(model=model, ledger=ledger)
    elif mode == "heuristic":
        judge = HeuristicJudge()
    elif mode == "both":
        h = check_answer(answer_path, sources_path, "heuristic",
                         out_prefix=None)
        l = check_answer(answer_path, sources_path, "llm", model=model,
                         top_k=top_k, out_prefix=out_prefix,
                         ledger_path=ledger_path)
        return {"heuristic": h, "llm": l}
    else:
        raise ValueError(f"unknown mode {mode!r}")
    verdicts = judge_claims(claims, chunks, judge, top_k=top_k)
    vd = [asdict(v) for v in verdicts]
    meta = {"answer": str(answer_path), "sources": str(sources_path),
            "mode": mode, "model": model if mode == "llm" else "-",
            "n_chunks": len(chunks)}
    if out_prefix:
        summary = write_reports(vd, meta, out_prefix)
    else:
        summary = summarize(vd)
    return {"meta": meta, "summary": summary, "verdicts": vd}


def validate_against_gold(gold_path: Path, sources_path: Path,
                          mode: str = "heuristic", model: str = "deepseek-chat",
                          top_k: int = 3, limit: int | None = None,
                          ledger_path: Path | None = None) -> dict:
    """Score a judge against the labelled gold set; returns confusion matrix."""
    gold = [json.loads(x) for x in
            Path(gold_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    if limit:
        gold = gold[:limit]
    chunks = load_sources(discover_sources(Path(sources_path)))
    if mode == "llm":
        from .judge import LLMJudge, Ledger
        ledger = Ledger(ledger_path or Path("results/cost_ledger.jsonl"))
        judge = LLMJudge(model=model, ledger=ledger)
    else:
        judge = HeuristicJudge()

    labels = ("supported", "unsupported", "contradicted")
    cm = {t: {p: 0 for p in labels} for t in labels}
    details = []
    for i, rec in enumerate(gold):
        hits = retrieve(rec["claim"], chunks, k=top_k)
        v = judge.judge(rec["claim"], [c for c, _ in hits])
        pred = v.verdict if v.verdict in labels else "unsupported"
        cm[rec["label"]][pred] += 1
        details.append({"id": rec["id"], "label": rec["label"], "pred": pred,
                        "score": v.score, "reason": v.reason,
                        "claim": rec["claim"][:80]})
        if (i + 1) % 50 == 0:
            print(f"[gold] {i + 1}/{len(gold)}")

    n = len(gold)
    correct = sum(cm[l][l] for l in labels)
    per_class = {}
    for l in labels:
        tp = cm[l][l]
        fn = sum(cm[l].values()) - tp
        fp = sum(cm[p][l] for p in labels if p != l)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec_ = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0
        per_class[l] = {"precision": round(prec, 4), "recall": round(rec_, 4),
                        "f1": round(f1, 4), "n": tp + fn}
    return {"mode": mode, "model": model if mode == "llm" else "-",
            "n": n, "accuracy": round(correct / n, 4) if n else 0.0,
            "confusion": cm, "per_class": per_class, "details": details}
