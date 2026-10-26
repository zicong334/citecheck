# -*- coding: utf-8 -*-
"""Report rendering: markdown + standalone HTML for a check run."""
from __future__ import annotations

import html
import json
from pathlib import Path

VERDICT_ZH = {"supported": "有据", "unsupported": "无据", "contradicted": "与源冲突"}
VERDICT_ICON = {"supported": "✅", "unsupported": "⚠️", "contradicted": "❌"}


def summarize(verdicts: list[dict]) -> dict:
    n = len(verdicts)
    counts = {v: sum(1 for x in verdicts if x["verdict"] == v)
              for v in ("supported", "unsupported", "contradicted")}
    return {
        "n_claims": n,
        **counts,
        "faithfulness": round(counts["supported"] / n, 4) if n else 0.0,
        "unsupported_rate": round(counts["unsupported"] / n, 4) if n else 0.0,
        "contradiction_rate": round(counts["contradicted"] / n, 4) if n else 0.0,
    }


def to_markdown(verdicts: list[dict], summary: dict, meta: dict) -> str:
    lines = ["# CiteCheck 忠实性报告", "",
             f"- 答案文件: `{meta.get('answer', '?')}`",
             f"- 证据源: `{meta.get('sources', '?')}`",
             f"- 判分模式: `{meta.get('mode', '?')}` / 模型: `{meta.get('model', '-')}`",
             "",
             f"**忠实性得分: {summary['faithfulness']:.1%}**"
             f"(有据 {summary['supported']} / 无据 {summary['unsupported']}"
             f" / 冲突 {summary['contradicted']},共 {summary['n_claims']} 条论断)",
             "", "| # | 判定 | 论断 | 证据 | 理由 |", "|---|---|---|---|---|"]
    for i, v in enumerate(verdicts, 1):
        claim = v["claim"].replace("|", "\\|")
        if len(claim) > 70:
            claim = claim[:70] + "…"
        ev = f"{v.get('evidence_doc', '')}#{v.get('evidence_idx', -1)}" \
            if v.get("evidence_idx", -1) >= 0 else "-"
        lines.append(f"| {i} | {VERDICT_ICON[v['verdict']]} {VERDICT_ZH[v['verdict']]} "
                     f"| {claim} | {ev} | {v.get('reason', '')} |")
    return "\n".join(lines) + "\n"


def to_html(verdicts: list[dict], summary: dict, meta: dict) -> str:
    rows = []
    for i, v in enumerate(verdicts, 1):
        cls = v["verdict"]
        rows.append(
            f'<tr class="{cls}"><td>{i}</td>'
            f'<td>{VERDICT_ICON[v["verdict"]]} {VERDICT_ZH[v["verdict"]]}</td>'
            f'<td>{html.escape(v["claim"])}</td>'
            f'<td>{html.escape(str(v.get("quote", "")))}</td>'
            f'<td>{html.escape(str(v.get("reason", "")))}</td></tr>')
    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CiteCheck 报告</title>
<style>
 body{{font-family:-apple-system,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;
      margin:2rem auto;max-width:1000px;color:#1a202c;line-height:1.55}}
 h1{{font-size:1.4rem}} .score{{font-size:2rem;font-weight:700}}
 table{{border-collapse:collapse;width:100%;font-size:.88rem}}
 td,th{{border:1px solid #e2e8f0;padding:.45rem .6rem;text-align:left;vertical-align:top}}
 tr.supported td:nth-child(2){{color:#22763d}}
 tr.unsupported{{background:#fffbeb}} tr.contradicted{{background:#fff5f5}}
 .meta{{color:#64748b;font-size:.85rem}}
</style></head><body>
<h1>CiteCheck 忠实性报告</h1>
<p class="meta">答案: {html.escape(str(meta.get('answer','?')))} ·
证据源: {html.escape(str(meta.get('sources','?')))} ·
模式: {html.escape(str(meta.get('mode','?')))}</p>
<p class="score">{summary['faithfulness']:.1%}</p>
<p>有据 {summary['supported']} · 无据 {summary['unsupported']} ·
冲突 {summary['contradicted']} · 共 {summary['n_claims']} 条论断</p>
<table><thead><tr><th>#</th><th>判定</th><th>论断</th><th>证据摘录</th><th>理由</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</body></html>"""


def write_reports(verdicts: list[dict], meta: dict, out_prefix: Path) -> dict:
    summary = summarize(verdicts)
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    (out_prefix.with_suffix(".json")).write_text(
        json.dumps({"meta": meta, "summary": summary, "verdicts": verdicts},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    (out_prefix.with_suffix(".md")).write_text(
        to_markdown(verdicts, summary, meta), encoding="utf-8")
    (out_prefix.with_suffix(".html")).write_text(
        to_html(verdicts, summary, meta), encoding="utf-8")
    return summary
