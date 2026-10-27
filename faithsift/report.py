# -*- coding: utf-8 -*-
"""报告渲染:markdown 与自包含 HTML。

两个渲染器都不依赖外部资源(HTML 把 CSS 内联),报告可以直接贴进 PR,
或从产物仓库里打开,不需要起静态服务器。
"""
from __future__ import annotations

import html
import json
from pathlib import Path

VERDICT_ZH = {"supported": "有据", "unsupported": "无据", "contradicted": "与源冲突"}
VERDICT_ICON = {"supported": "✅", "unsupported": "⚠️", "contradicted": "❌"}
VERDICT_COLOR = {"supported": "#22763d", "unsupported": "#b45309",
                 "contradicted": "#b91c1c"}


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
        "multi_hop": sum(1 for x in verdicts if x.get("multi_hop")),
    }


def evidence_label(v: dict) -> str:
    """``doc#3`` for single-chunk evidence, ``doc#3+7`` for a multi-hop chain."""
    idxs = v.get("evidence_idxs") or []
    if not idxs and v.get("evidence_idx", -1) >= 0:
        idxs = [v["evidence_idx"]]
    if not idxs:
        return "-"
    doc = str(v.get("evidence_doc", "") or "?")
    return f"{doc}#" + "+".join(str(i) for i in idxs)


def _problem_rows(verdicts: list[dict]) -> list[dict]:
    return [v for v in verdicts if v["verdict"] != "supported"]


def to_markdown(verdicts: list[dict], summary: dict, meta: dict) -> str:
    hop = f",多跳 {summary.get('multi_hop', 0)}" if summary.get("multi_hop") else ""
    lines = ["# FaithSift 忠实性报告", "",
             f"- 答案文件: `{meta.get('answer', '?')}`",
             f"- 证据源: `{meta.get('sources', '?')}`",
             f"- 判分模式: `{meta.get('mode', '?')}` / 模型: `{meta.get('model', '-')}`",
             f"- 检索通道: `{meta.get('retriever', 'lexical')}` "
             f"/ 向量模型: `{meta.get('embedder', '-')}`",
             "",
             f"**忠实性得分: {summary['faithfulness']:.1%}**"
             f"(有据 {summary['supported']} / 无据 {summary['unsupported']}"
             f" / 冲突 {summary['contradicted']},共 {summary['n_claims']} 条论断{hop})",
             "", "| # | 判定 | 论断 | 证据 | 理由 |", "|---|---|---|---|---|"]
    for i, v in enumerate(verdicts, 1):
        claim = v["claim"].replace("|", "\\|")
        if len(claim) > 70:
            claim = claim[:70] + "…"
        tag = " 🔗" if v.get("multi_hop") else ""
        lines.append(f"| {i} | {VERDICT_ICON[v['verdict']]} {VERDICT_ZH[v['verdict']]} "
                     f"| {claim} | {evidence_label(v)}{tag} "
                     f"| {str(v.get('reason', '')).replace('|', '/')} |")
    problems = _problem_rows(verdicts)
    if problems:
        lines += ["", "## 待处理论断", ""]
        for v in problems:
            lines.append(f"- {VERDICT_ICON[v['verdict']]} **{VERDICT_ZH[v['verdict']]}** "
                         f"{v['claim']} — {v.get('reason', '')}")
    return "\n".join(lines) + "\n"


_CSS = """
:root{color-scheme:light}
*{box-sizing:border-box}
body{font-family:-apple-system,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;
     margin:2rem auto;max-width:1080px;padding:0 1rem;color:#1a202c;line-height:1.55;
     background:#fbfbfd}
h1{font-size:1.45rem;margin:0 0 .3rem}
h2{font-size:1.05rem;margin:1.8rem 0 .6rem}
.sub{color:#64748b;font-size:.85rem;margin:.2rem 0}
.score{font-size:2.6rem;font-weight:700;margin:.4rem 0;letter-spacing:-.02em}
.bar{height:10px;border-radius:6px;background:#e2e8f0;overflow:hidden;display:flex}
.bar span{display:block;height:100%}
.chips{display:flex;gap:.5rem;flex-wrap:wrap;margin:.7rem 0 0}
.chip{font-size:.8rem;padding:.22rem .6rem;border-radius:999px;background:#eef2f7;color:#334155}
.chip.ok{background:#e7f6ec;color:#22763d}
.chip.warn{background:#fef3c7;color:#92400e}
.chip.bad{background:#fee2e2;color:#b91c1c}
table{border-collapse:collapse;width:100%;font-size:.86rem;background:#fff;
      border-radius:10px;overflow:hidden;box-shadow:0 1px 2px rgba(15,23,42,.06)}
td,th{border-bottom:1px solid #eef2f7;padding:.5rem .65rem;text-align:left;vertical-align:top}
th{background:#f8fafc;font-weight:600;color:#475569;font-size:.8rem;
   text-transform:uppercase;letter-spacing:.03em}
tr.supported td:nth-child(2){color:#22763d;font-weight:600}
tr.unsupported{background:#fffbeb}
tr.contradicted{background:#fff5f5}
tr.contradicted td:nth-child(2){color:#b91c1c;font-weight:600}
td.c{white-space:nowrap;color:#64748b;font-variant-numeric:tabular-nums}
code{background:#f1f5f9;padding:.1rem .3rem;border-radius:4px;font-size:.85em}
.hop{font-size:.7rem;border:1px solid #c7d2fe;color:#4338ca;border-radius:4px;
     padding:0 .25rem;margin-left:.25rem;white-space:nowrap}
.empty{color:#64748b;font-size:.88rem}
"""


def to_html(verdicts: list[dict], summary: dict, meta: dict) -> str:
    n = summary["n_claims"] or 1
    pct = {k: summary[k] / n * 100 for k in
           ("supported", "unsupported", "contradicted")}
    rows = []
    for i, v in enumerate(verdicts, 1):
        cls = v["verdict"]
        hop = '<span class="hop">多跳</span>' if v.get("multi_hop") else ""
        rows.append(
            f'<tr class="{cls}"><td class="c">{i}</td>'
            f'<td>{VERDICT_ICON[v["verdict"]]} {VERDICT_ZH[v["verdict"]]}</td>'
            f'<td>{html.escape(v["claim"])}</td>'
            f'<td>{html.escape(evidence_label(v))}{hop}</td>'
            f'<td>{html.escape(str(v.get("quote", "")))}</td>'
            f'<td>{html.escape(str(v.get("reason", "")))}</td></tr>')

    problems = _problem_rows(verdicts)
    if problems:
        items = "".join(
            f'<li><b style="color:{VERDICT_COLOR[v["verdict"]]}">'
            f'{VERDICT_ZH[v["verdict"]]}</b> {html.escape(v["claim"])}'
            f'<br><span class="sub">{html.escape(str(v.get("reason", "")))}</span></li>'
            for v in problems)
        problem_block = (f'<h2>待处理论断({len(problems)})</h2>'
                         f'<ul style="font-size:.9rem;line-height:1.7">{items}</ul>')
    else:
        problem_block = '<h2>待处理论断</h2><p class="empty">无 — 全部论断均有据。</p>'

    meta_bits = [
        f"答案: {html.escape(str(meta.get('answer', '?')))}",
        f"证据源: {html.escape(str(meta.get('sources', '?')))}",
        f"模式: {html.escape(str(meta.get('mode', '?')))}",
        f"模型: {html.escape(str(meta.get('model', '-')))}",
        f"检索: {html.escape(str(meta.get('retriever', 'lexical')))}",
    ]
    if meta.get("embedder") and meta.get("embedder") != "-":
        meta_bits.append(f"向量: {html.escape(str(meta['embedder']))}")

    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FaithSift 报告 · {summary['faithfulness']:.1%}</title>
<style>{_CSS}</style></head><body>
<h1>FaithSift 忠实性报告</h1>
<p class="sub">{' · '.join(meta_bits)}</p>
<p class="score" style="color:{_score_color(summary['faithfulness'])}">
{summary['faithfulness']:.1%}</p>
<div class="bar" role="img" aria-label="有据 / 无据 / 冲突 占比">
 <span style="width:{pct['supported']:.2f}%;background:#22c55e"></span>
 <span style="width:{pct['unsupported']:.2f}%;background:#f59e0b"></span>
 <span style="width:{pct['contradicted']:.2f}%;background:#ef4444"></span>
</div>
<div class="chips">
 <span class="chip ok">✅ 有据 {summary['supported']}</span>
 <span class="chip warn">⚠️ 无据 {summary['unsupported']}</span>
 <span class="chip bad">❌ 冲突 {summary['contradicted']}</span>
 <span class="chip">共 {summary['n_claims']} 条论断</span>
{f' <span class="chip">🔗 多跳 {summary["multi_hop"]}</span>' if summary.get("multi_hop") else ''}
</div>
{problem_block}
<h2>逐条判定</h2>
<table><thead><tr><th>#</th><th>判定</th><th>论断</th><th>证据</th>
<th>证据摘录</th><th>理由</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p class="sub">由 FaithSift 生成 · 判定标准: supported=证据明确包含 /
contradicted=同一事实取值冲突 / unsupported=证据沉默</p>
</body></html>"""


def _score_color(score: float) -> str:
    if score >= 0.9:
        return "#22763d"
    if score >= 0.7:
        return "#b45309"
    return "#b91c1c"


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


__all__ = ["VERDICT_ZH", "evidence_label", "summarize", "to_html",
           "to_markdown", "write_reports"]
