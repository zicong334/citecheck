# -*- coding: utf-8 -*-
"""命令行入口:``python -m faithsift <子命令> ...``

子命令
------
``check <answer> --sources <dir>``
    逐条判定一份回答的论断是否有据。
    ``--mode heuristic|llm|both``、``--retriever lexical|hybrid|dense``、
    ``--embedder hashing|<模型名>``、``--top-k N``、``--out results/check``。
    CI 门禁:``--fail-under 0.9`` 与 ``--fail-on-contradicted``。

``gold``
    重建确定性金标(``gold/gold.jsonl`` 与 ``gold/hard_gold.jsonl``)。

``validate``
    用金标集给判官打分。``--dataset gold|hard`` 选择数据集;
    ``--min-accuracy`` 把它变成 CI 门禁。

退出码
------
``0`` 成功 · ``2`` 用法错误 · ``3`` 忠实性低于 ``--fail-under``
· ``4`` 配合 ``--fail-on-contradicted`` 发现与源冲突的论断
· ``5`` 金标准确率低于 ``--min-accuracy`` · ``6`` 缺少密钥等凭据。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GATES = {"fail_under": 3, "contradicted": 4, "accuracy": 5}


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="faithsift",
        description="面向 RAG 回答的句子级引用忠实性检测器。")
    ap.add_argument("--version", action="version", version="faithsift 0.3.0")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="核查一份回答对证据源的忠实性")
    c.add_argument("answer", help="被核验的回答文件(md/txt)")
    c.add_argument("--sources", required=True, help="证据源目录或单个文件")
    c.add_argument("--mode", default="heuristic",
                   choices=["heuristic", "llm", "both"],
                   help="判官:heuristic=离线规则,llm=DeepSeek,both=两者都跑")
    c.add_argument("--model", default="deepseek-chat", help="LLM 判官模型名")
    c.add_argument("--retriever", default="lexical",
                   choices=["lexical", "hybrid", "dense"],
                   help="检索通道:lexical=BM25+数值(离线);hybrid/dense 需配 --embedder")
    c.add_argument("--embedder", default=None,
                   help="向量后端:'hashing'(离线替身)、'local'(本地离线 ONNX 多语模型,"
                        "需 pip install 'faithsift[semantic]')、'local:<hf-model-id>',"
                        "或 OpenAI 兼容的嵌入模型名")
    c.add_argument("--top-k", type=int, default=3, help="每条论断取多少证据片段")
    c.add_argument("--out", default="results/check",
                   help="报告前缀;会写出 .md/.html/.json 三份")
    c.add_argument("--no-report", action="store_true",
                   help="只打印摘要,不写文件")
    c.add_argument("--fail-under", type=float, default=None,
                   help="忠实性低于该比例时以退出码 3 结束")
    c.add_argument("--fail-on-contradicted", action="store_true",
                   help="存在与源冲突的论断时以退出码 4 结束")
    c.add_argument("--json", action="store_true", help="输出机器可读 JSON")

    g = sub.add_parser("gold", help="重建 gold.jsonl 与 hard_gold.jsonl")
    g.add_argument("--gold-dir", default=None, help="金标目录(默认 <仓库>/gold)")

    v = sub.add_parser("validate", help="用金标集给判官打分")
    v.add_argument("--dataset", default="gold", choices=["gold", "hard"],
                   help="gold=合成金标(回归下限),hard=人工困难金标")
    v.add_argument("--mode", default="heuristic",
                   choices=["heuristic", "llm"], help="用哪个判官打分")
    v.add_argument("--model", default="deepseek-chat", help="LLM 判官模型名")
    v.add_argument("--retriever", default="lexical",
                   choices=["lexical", "hybrid", "dense"], help="检索通道")
    v.add_argument("--embedder", default=None, help="向量后端")
    v.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    v.add_argument("--gold", default=None, help="自定义金标文件路径")
    v.add_argument("--sources", default=None, help="自定义证据源目录")
    v.add_argument("--top-k", type=int, default=3, help="每条论断取多少证据片段")
    v.add_argument("--min-accuracy", type=float, default=None,
                   help="准确率低于该比例时以退出码 5 结束")
    v.add_argument("--out", default=None, help="完整结果 JSON 的写出路径")
    return ap


def _cmd_gold(args: argparse.Namespace) -> int:
    gold_dir = Path(args.gold_dir) if args.gold_dir else ROOT / "gold"
    sys.path.insert(0, str(gold_dir.parent))
    import importlib.util

    def run(name: str) -> int:
        path = gold_dir / f"{name}.py"
        if not path.exists():
            print(f"[gold] skip missing {path}", file=sys.stderr)
            return 0
        spec = importlib.util.spec_from_file_location(name + "_mod", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.main()

    rc = run("build_gold")
    rc = run("build_hard_gold") or rc
    return rc


def _cmd_check(args: argparse.Namespace) -> int:
    from .pipeline import check_answer, gate_failures

    try:
        res = check_answer(Path(args.answer), Path(args.sources),
                           mode=args.mode, model=args.model,
                           top_k=args.top_k,
                           out_prefix=None if args.no_report else Path(args.out),
                           retriever=args.retriever, embedder=args.embedder)
    except RuntimeError as e:  # missing credentials / backend
        print(f"faithsift: {e}", file=sys.stderr)
        return 6

    if args.mode == "both":
        summary = res["llm"]["summary"]
    else:
        summary = res["summary"]

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
    else:
        _print_summary(summary, res, args)

    problems = gate_failures(summary, args.fail_under, args.fail_on_contradicted)
    if problems:
        for p in problems:
            print(f"GATE FAIL: {p}", file=sys.stderr)
        rc = (GATES["contradicted"]
              if any("contradicted" in p for p in problems) and args.fail_on_contradicted
              else GATES["fail_under"])
        return rc
    return 0


def _print_summary(summary: dict, res: dict, args: argparse.Namespace) -> None:
    print(f"faithfulness: {summary['faithfulness']:.1%}"
          f"  (supported {summary['supported']} / unsupported "
          f"{summary['unsupported']} / contradicted {summary['contradicted']}"
          f" of {summary['n_claims']} claims)")
    if summary.get("multi_hop"):
        print(f"multi-hop claims: {summary['multi_hop']}")
    block = res.get("llm") or res
    for v in block.get("verdicts", []):
        if v["verdict"] != "supported":
            print(f"  [{v['verdict']:12s}] {v['claim'][:70]}")
            print(f"                 -> {v.get('reason', '')}")
    if not args.no_report:
        print(f"reports -> {args.out}.{{md,html,json}}")


def _cmd_validate(args: argparse.Namespace) -> int:
    from .pipeline import validate_against_gold

    gold = Path(args.gold) if args.gold else \
        ROOT / "gold" / ("hard_gold.jsonl" if args.dataset == "hard" else "gold.jsonl")
    if not gold.exists():
        print(f"faithsift: missing gold set {gold}; run `faithsift gold`",
              file=sys.stderr)
        return 2
    try:
        res = validate_against_gold(
            gold, Path(args.sources or ROOT / "gold" / "sources"),
            mode=args.mode, model=args.model, limit=args.limit,
            top_k=args.top_k, retriever=args.retriever, embedder=args.embedder,
            progress=args.limit is None or args.limit > 50)
    except RuntimeError as e:
        print(f"faithsift: {e}", file=sys.stderr)
        return 6

    out = Path(args.out) if args.out else \
        ROOT / "results" / f"gold_validation_{args.dataset}_{args.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    summary = {k: res[k] for k in ("mode", "model", "retriever", "gold", "n",
                                   "accuracy", "per_class", "by_category",
                                   "multi_hop")}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"full -> {out}")
    if args.min_accuracy is not None and res["accuracy"] < args.min_accuracy:
        print(f"GATE FAIL: accuracy {res['accuracy']:.1%} < "
              f"{args.min_accuracy:.1%}", file=sys.stderr)
        return GATES["accuracy"]
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "gold":
        return _cmd_gold(args)
    if args.cmd == "check":
        return _cmd_check(args)
    if args.cmd == "validate":
        return _cmd_validate(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
