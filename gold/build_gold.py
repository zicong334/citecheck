# -*- coding: utf-8 -*-
"""Deterministic gold-set builder.

Three label classes generated from the source corpus:
  supported     extracted verbatim sentences (facts present in sources)
  contradicted  sentences with one distinctive number altered to a value
                absent from the whole corpus (direct conflict with source)
  unsupported   template claims about corpus entities on topics the corpus
                never addresses (budgets, heritage listings...) -> no evidence
                either way. Topic absence is asserted by a guard list checked
                against the corpus at build time.

Output: gold/gold.jsonl  {"id","claim","label","source_doc","transform"}
Usage:  python gold/build_gold.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from citecheck.claims import split_claims  # noqa: E402

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "sources"
OUT = ROOT / "gold.jsonl"
SEED = 20261027

NUM = re.compile(r"\d+(?:\.\d+)?")

# topics guaranteed absent from every source doc (guard-checked below)
ABSENT_GUARD = ["预算", "世界文化遗产", "UNESCO", "maintenance cost",
                "heritage", "budget", "美元", "USD"]

ZH_SUBJECTS = ["茶树", "阿姆斯特丹", "养蜂业", "睡眠研究", "城市绿地", "咖啡烘焙"]
EN_SUBJECTS = ["coffee roasting", "honeybees", "Amsterdam", "sleep research",
               "green roofs", "tea cultivation"]

ZH_TEMPLATES = [
    "{s}在2025年的相关研发预算约为{n}万美元。",
    "{s}于2019年被联合国教科文组织列入世界文化遗产名录。",
    "{s}的年度维护成本在2024年达到了{n}万元。",
]
EN_TEMPLATES = [
    "The annual maintenance cost of {s} reached USD {n} million in 2024.",
    "{s} was listed as a UNESCO World Heritage site in 2019.",
    "The research budget for {s} exceeded USD {n} million in 2025.",
]


def corpus_text() -> str:
    parts = []
    for fp in sorted(SOURCES.glob("*.md")):
        parts.append(fp.read_text(encoding="utf-8"))
    return "\n".join(parts)


def altered_number(sentence: str, corpus: str, rng: random.Random) -> str | None:
    nums = [m for m in NUM.finditer(sentence)
            if len(m.group()) >= 2 or "." in m.group()]
    if not nums:
        return None
    m = rng.choice(nums)
    old = m.group()
    base = int(float(old))
    for _ in range(50):
        cand = str(base + rng.choice([37, 53, 71, 113, 991]))
        if "." in old:
            cand = f"{float(old) + rng.choice([3.7, 11.3, 53.5]):.1f}"
        if cand not in corpus and cand not in sentence:
            return sentence[:m.start()] + cand + sentence[m.end():]
    return None


def main() -> int:
    rng = random.Random(SEED)
    corpus = corpus_text()
    for g in ABSENT_GUARD:
        assert g.lower() not in corpus.lower(), f"guard violated: {g!r} in corpus"

    records: list[dict] = []

    # 1) supported: extracted sentences
    per_doc: dict[str, list[str]] = {}
    for fp in sorted(SOURCES.glob("*.md")):
        sentences = split_claims(fp.read_text(encoding="utf-8"))
        per_doc[fp.stem] = sentences
        for s in sentences:
            records.append({"claim": s, "label": "supported",
                            "source_doc": fp.name, "transform": "extract"})

    # 2) contradicted: number-altered sentences (up to 12 per doc)
    for doc, sentences in per_doc.items():
        cands = [s for s in sentences if NUM.search(s)]
        rng.shuffle(cands)
        made = 0
        for s in cands:
            if made >= 12:
                break
            alt = altered_number(s, corpus, rng)
            if alt and alt != s:
                records.append({"claim": alt, "label": "contradicted",
                                "source_doc": doc + ".md",
                                "transform": "number_alter"})
                made += 1

    # 3) unsupported: template claims on absent topics (2 numeric variants each)
    for tpl in ZH_TEMPLATES:
        for s in ZH_SUBJECTS:
            for _ in range(2):
                n = rng.randint(120, 9800)
                records.append({"claim": tpl.format(s=s, n=n),
                                "label": "unsupported", "source_doc": "",
                                "transform": "absent_topic_template"})
    for tpl in EN_TEMPLATES:
        for s in EN_SUBJECTS:
            for _ in range(2):
                n = rng.randint(120, 9800)
                records.append({"claim": tpl.format(s=s, n=n),
                                "label": "unsupported", "source_doc": "",
                                "transform": "absent_topic_template"})

    rng.shuffle(records)
    for i, r in enumerate(records):
        r["id"] = f"G{i + 1:04d}"
        r = {k: r[k] for k in ("id", "claim", "label", "source_doc", "transform")}
        records[i] = r

    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records),
                   encoding="utf-8")
    from collections import Counter
    dist = Counter(r["label"] for r in records)
    print(f"gold -> {OUT}: {len(records)} claims {dict(dist)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
