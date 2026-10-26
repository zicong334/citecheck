# -*- coding: utf-8 -*-
"""CLI: python -m citecheck <command> ...

Commands:
  check     <answer.md> --sources <dir> [--mode heuristic|llm|both]
            [--model deepseek-chat] [--top-k 3] [--out results/check]
  gold      build the gold set (runs gold/build_gold.py logic)
  validate  score a judge against the gold set
            [--mode heuristic|llm] [--limit N] [--out results/gold_validation.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="citecheck")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="check an answer file against sources")
    c.add_argument("answer")
    c.add_argument("--sources", required=True)
    c.add_argument("--mode", default="heuristic",
                   choices=["heuristic", "llm", "both"])
    c.add_argument("--model", default="deepseek-chat")
    c.add_argument("--top-k", type=int, default=3)
    c.add_argument("--out", default="results/check")

    g = sub.add_parser("gold", help="rebuild gold/gold.jsonl")
    g.add_argument("--gold-dir", default=None)

    v = sub.add_parser("validate", help="score judge against gold set")
    v.add_argument("--mode", default="heuristic", choices=["heuristic", "llm"])
    v.add_argument("--model", default="deepseek-chat")
    v.add_argument("--limit", type=int, default=None)
    v.add_argument("--gold", default=None)
    v.add_argument("--sources", default=None)
    v.add_argument("--top-k", type=int, default=3)
    v.add_argument("--out", default=None)

    args = ap.parse_args(argv)

    if args.cmd == "gold":
        gold_dir = Path(args.gold_dir) if args.gold_dir else \
            Path(__file__).resolve().parent.parent / "gold"
        sys.path.insert(0, str(gold_dir.parent))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "build_gold", gold_dir / "build_gold.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.main()

    if args.cmd == "check":
        from .pipeline import check_answer
        res = check_answer(Path(args.answer), Path(args.sources),
                           mode=args.mode, model=args.model,
                           top_k=args.top_k, out_prefix=Path(args.out))
        s = res.get("summary") or res.get("llm", {}).get("summary")
        print(json.dumps(s, ensure_ascii=False, indent=1))
        return 0

    if args.cmd == "validate":
        root = Path(__file__).resolve().parent.parent
        from .pipeline import validate_against_gold
        res = validate_against_gold(
            Path(args.gold or root / "gold" / "gold.jsonl"),
            Path(args.sources or root / "gold" / "sources"),
            mode=args.mode, model=args.model, limit=args.limit,
            top_k=args.top_k)
        out = Path(args.out) if args.out else \
            root / "results" / f"gold_validation_{args.mode}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        summary = {k: res[k] for k in ("mode", "model", "n", "accuracy",
                                       "per_class")}
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        print(f"full -> {out}")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
