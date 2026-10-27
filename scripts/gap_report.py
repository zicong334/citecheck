# -*- coding: utf-8 -*-
"""开发辅助:按数据集打印逐条错分报告。

用法:
    python scripts/gap_report.py --dataset hard [--mode heuristic] [--limit N]

输出准确率、分类准确率、多跳召回,以及每一条判错的论断和它被判错的理由——
比只看一个总分更容易看出**哪项能力**缺了。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from citecheck.pipeline import validate_against_gold  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hard", choices=["gold", "hard", "custom"])
    ap.add_argument("--gold", default=None)
    ap.add_argument("--mode", default="heuristic", choices=["heuristic", "llm"])
    ap.add_argument("--retriever", default="lexical",
                    choices=["lexical", "hybrid", "dense"])
    ap.add_argument("--embedder", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only-errors", action="store_true")
    args = ap.parse_args()

    gold = Path(args.gold) if args.gold else ROOT / "gold" / (
        "hard_gold.jsonl" if args.dataset == "hard" else "gold.jsonl")
    res = validate_against_gold(
        gold, ROOT / "gold" / "sources", mode=args.mode,
        retriever=args.retriever, embedder=args.embedder,
        limit=args.limit, progress=False)

    print(f"gold={gold.name} mode={res['mode']} retriever={res['retriever']} "
          f"n={res['n']} accuracy={res['accuracy']:.2%}")
    print("per-class F1: " + ", ".join(
        f"{k} {v['f1']:.2f} (n={v['n']})" for k, v in res["per_class"].items()))
    if res.get("multi_hop", {}).get("expected"):
        print(f"multi-hop recall: {res['multi_hop']['recall']:.0%} "
              f"({res['multi_hop']['detected']}/{res['multi_hop']['expected']})")
    print("per-category accuracy:")
    for cat, st in res["by_category"].items():
        print(f"  {cat:18s} {st['accuracy']:6.1%}  n={st['n']}")
    errors = [d for d in res["details"] if d["pred"] != d["label"]]
    print(f"errors: {len(errors)}/{res['n']}")
    for d in errors:
        print(f"  {d['id']} [{d['category']}] {d['label']} -> {d['pred']}: "
              f"{d['claim'][:52]}")
        print(f"        reason: {d['reason']}")
    Path("results").mkdir(exist_ok=True)
    out = Path("results") / f"gap_{args.dataset}_{args.mode}_{args.retriever}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"detail -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
