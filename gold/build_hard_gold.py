# -*- coding: utf-8 -*-
"""Hand-authored *hard* gold set (lexically unfriendly, human-labelled).

Why a second gold set
---------------------
``build_gold.py`` produces 184 claims by *transforming* source sentences
(extract / number-alter / absent-topic template). That set is deterministic and
great as a regression floor, but it is lexical-friendly: the supported class is
verbatim, the contradicted class differs by one token, and the unsupported class
is template-shaped. A high score there does not transfer to real RAG traffic.

This set is the honest counterweight: 49 hand-written claims against the same
six-document corpus, each with a rationale, targeting the failure modes that
actually show up in production:

===============  =============================================================
category         what it probes
===============  =============================================================
paraphrase_zh    synonym/word-order rewrite, no shared rare tokens
crosslingual     zh claim vs en source (and vice versa) with **no digits**, so
                 only the bilingual concept channel can retrieve them
multi_hop        the fact needs two paragraphs (``hops`` = 2)
number_alter     a distinctive number changed
direction        same numbers, but 升高/降低 (or 高于/低于) flipped
entity_swap      a fact attached to the wrong entity (张冠李戴)
unit             right fact, wrong unit/quantity
plausible_absent plausible-sounding fact the corpus never states
near_miss        corpus discusses the adjacent subtopic, not this fact
===============  =============================================================

Labels follow the three definitions in README: ``contradicted`` requires the
sources to state something *conflicting*; ``unsupported`` requires silence.
Several cases are deliberately hostile to the heuristic engine — the point is to
measure the gap, not to flatter it.

Output: ``gold/hard_gold.jsonl`` (id, claim, label, category, hops,
evidence_docs, rationale). Deterministic: fixed case list + seeded shuffle.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from faithsift.evidence import distinctive_numbers  # noqa: E402

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "sources"
OUT = ROOT / "hard_gold.jsonl"
SEED = 20261028

# id is assigned after shuffling; keep the authored order for readability.
CASES: list[dict] = [
    # ------------------------------------------------------------------
    # supported / paraphrase_zh — rewritten, not extracted
    # ------------------------------------------------------------------
    {"claim": "中国西南地区是茶树的原产地,当地种茶的历史已经超过两千年。",
     "label": "supported", "category": "paraphrase_zh", "hops": 1,
     "evidence_docs": ["tea_zh.md"],
     "rationale": "源文: 茶树原产于中国西南部,栽培历史超过 2000 年。同义改写。"},
    {"claim": "绿茶加工要用高温杀青让多酚氧化酶失去活性,茶汤才能保持清亮。",
     "label": "supported", "category": "paraphrase_zh", "hops": 1,
     "evidence_docs": ["tea_zh.md"],
     "rationale": "源文: 杀青温度 200 度以上,高温钝化多酚氧化酶,保持绿叶清汤。"},
    {"claim": "红茶发酵时儿茶素会转成茶黄素和茶红素,其中茶黄素决定茶汤的明亮度。",
     "label": "supported", "category": "paraphrase_zh", "hops": 1,
     "evidence_docs": ["tea_zh.md"],
     "rationale": "源文: 儿茶素生成茶黄素和茶红素,茶黄素是汤色明亮度的关键指标。"},
    {"claim": "乌龙茶只做部分发酵,做青会让叶片边缘破损,形成绿叶带红边的样子。",
     "label": "supported", "category": "paraphrase_zh", "hops": 1,
     "evidence_docs": ["tea_zh.md"],
     "rationale": "源文: 乌龙茶半发酵,做青使叶片边缘轻微破损,绿叶红镶边。"},
    {"claim": "成年人一夜大约会经历四到六个睡眠周期,每个周期九十分钟上下。",
     "label": "supported", "category": "paraphrase_zh", "hops": 1,
     "evidence_docs": ["sleep_zh.md"],
     "rationale": "源文: 4 至 6 个周期,每个约 90 分钟;数字写作汉字。"},
    {"claim": "长期睡不够的人往往低估自己到底有多困。",
     "label": "supported", "category": "paraphrase_zh", "hops": 1,
     "evidence_docs": ["sleep_zh.md"],
     "rationale": "源文: 慢性睡眠不足者会低估自己的受损程度。"},

    # ------------------------------------------------------------------
    # supported / crosslingual — zh claim, en source, no digits
    # ------------------------------------------------------------------
    {"claim": "咖啡生豆在烘焙前含有相当比例的水分,烘焙结束时基本被烘干。",
     "label": "supported", "category": "crosslingual", "hops": 1,
     "evidence_docs": ["coffee_en.md"],
     "rationale": "coffee_en: green beans 8-12% moisture drops to 1-2% by the end of "
                  "roasting. 中文论断、英文源、无阿拉伯数字。"},
    {"claim": "深度烘焙的咖啡豆按体积算,咖啡因反而略少于浅烘。",
     "label": "supported", "category": "crosslingual", "hops": 1,
     "evidence_docs": ["coffee_en.md"],
     "rationale": "coffee_en: darker roasts contain slightly less caffeine by volume."},
    {"claim": "美拉德反应会生成数百种香气物质,也让咖啡豆变成褐色。",
     "label": "supported", "category": "crosslingual", "hops": 1,
     "evidence_docs": ["coffee_en.md"],
     "rationale": "coffee_en: Maillard reaction produces several hundred aromatic "
                  "compounds and the characteristic brown color."},
    {"claim": "蜜蜂用摇摆舞把食物所在的方向和距离告诉同伴。",
     "label": "supported", "category": "crosslingual", "hops": 1,
     "evidence_docs": ["bees_en.md"],
     "rationale": "bees_en: waggle dance encodes direction (angle) and distance."},
    {"claim": "蜂王在春季高峰期每天能产下两千粒卵。",
     "label": "supported", "category": "crosslingual", "hops": 1,
     "evidence_docs": ["bees_en.md"],
     "rationale": "bees_en: queen lays up to 2,000 eggs per day during spring peak."},

    # ------------------------------------------------------------------
    # supported / crosslingual — en claim, zh source
    # ------------------------------------------------------------------
    {"claim": "Tea bushes do best where the annual average temperature stays between "
              "fifteen and twenty-five degrees Celsius.",
     "label": "supported", "category": "crosslingual", "hops": 1,
     "evidence_docs": ["tea_zh.md"],
     "rationale": "tea_zh: 年均温度 15 至 25 摄氏度;英文论断无阿拉伯数字。"},
    {"claim": "Copenhagen has invested heavily in cycling infrastructure for years, "
              "and roughly half of its commuters now cycle to work or school.",
     "label": "supported", "category": "crosslingual", "hops": 1,
     "evidence_docs": ["cycling_zh.md"],
     "rationale": "cycling_zh: 哥本哈根长期投资自行车基础设施,通勤骑车比例约 49%。"},
    {"claim": "Urban tree canopy cools the ground far more effectively than plain grass.",
     "label": "supported", "category": "crosslingual", "hops": 1,
     "evidence_docs": ["greenspace_zh.md"],
     "rationale": "greenspace_zh: 树冠遮阴降温 10 至 25 摄氏度,比草地更强。"},

    # ------------------------------------------------------------------
    # supported / multi_hop — one claim, two paragraphs
    # ------------------------------------------------------------------
    {"claim": "茶树喜欢 15 至 25 摄氏度的年均气温,而制作绿茶时杀青温度要升到 "
              "200 摄氏度以上。",
     "label": "supported", "category": "multi_hop", "hops": 2,
     "evidence_docs": ["tea_zh.md"],
     "rationale": "第 1 段给年均温,第 2 段给杀青温度,需两段合并。"},
    {"claim": "阿姆斯特丹约有 38% 的出行靠自行车,哥本哈根通勤者里骑车上班上学的"
              "约占 49%。",
     "label": "supported", "category": "multi_hop", "hops": 2,
     "evidence_docs": ["cycling_zh.md"],
     "rationale": "两段各给一个城市的数据,跨段落(同文档)多跳。"},
    {"claim": "城市树冠遮阴能让地表降温 10 至 25 摄氏度,而屋顶绿化只让顶层室内"
              "降温 2 至 4 摄氏度。",
     "label": "supported", "category": "multi_hop", "hops": 2,
     "evidence_docs": ["greenspace_zh.md"],
     "rationale": "第 2 段树冠降温,第 4 段屋顶绿化降温。"},
    {"claim": "成人每夜约经历 4 至 6 个睡眠周期,而针对慢性失眠的 CBT-I 通常要持续 "
              "6 至 8 周。",
     "label": "supported", "category": "multi_hop", "hops": 2,
     "evidence_docs": ["sleep_zh.md"],
     "rationale": "第 1 段睡眠周期,第 5 段 CBT-I 疗程。"},
    {"claim": "红茶里的茶黄素一般占干重 0.5% 至 2%,而中国 2023 年的茶叶产量约为 "
              "355 万吨。",
     "label": "supported", "category": "multi_hop", "hops": 2,
     "evidence_docs": ["tea_zh.md"],
     "rationale": "第 3 段茶黄素比例,第 4 段产量。"},
    {"claim": "生产一公斤蜂蜜需要蜜蜂访问约 10 万朵花,而全球约 75% 的主要农作物"
              "类型受益于动物授粉。",
     "label": "supported", "category": "multi_hop", "hops": 2,
     "evidence_docs": ["bees_en.md"],
     "rationale": "bees_en 第 3 段(花数)与第 5 段(75% 作物)合并。"},
    {"claim": "隔离自行车道可把严重伤害事故降低约 60% 至 75%,而北京回龙观自行车"
              "专用路全长约 6.5 公里。",
     "label": "supported", "category": "multi_hop", "hops": 2,
     "evidence_docs": ["cycling_zh.md"],
     "rationale": "第 3 段隔离车道效果,第 4 段北京专用路长度。"},

    # ------------------------------------------------------------------
    # supported / verbatim — sanity anchors
    # ------------------------------------------------------------------
    {"claim": "茶多酚占干重的 18% 至 36%,其中儿茶素类约占茶多酚总量的 60% 至 80%。",
     "label": "supported", "category": "verbatim", "hops": 1,
     "evidence_docs": ["tea_zh.md"], "rationale": "原文抽取。"},
    {"claim": "浓缩咖啡通常在 9 bar 压力下萃取,水温保持在 90 至 96 摄氏度。",
     "label": "supported", "category": "verbatim", "hops": 1,
     "evidence_docs": ["coffee_en.md"], "rationale": "coffee_en 第 5 段原文。"},
    {"claim": "全球天然蜂蜜产量在 2022 年约为 190 万吨。",
     "label": "supported", "category": "verbatim", "hops": 1,
     "evidence_docs": ["bees_en.md"],
     "rationale": "bees_en 第 3 段原文(1.9 million tonnes);量级换算后与 190 万吨一致。"},
    {"claim": "佛得草式轻型屋顶绿化的基质厚度通常为 8 至 15 厘米,荷载低于每平方米 "
              "150 公斤。",
     "label": "supported", "category": "verbatim", "hops": 1,
     "evidence_docs": ["greenspace_zh.md"], "rationale": "第 4 段原文。"},

    # ------------------------------------------------------------------
    # contradicted / number_alter
    # ------------------------------------------------------------------
    {"claim": "茶树适宜生长的年均温度为 16 至 25 摄氏度。",
     "label": "contradicted", "category": "number_alter", "hops": 1,
     "evidence_docs": ["tea_zh.md"],
     "rationale": "源文 15 至 25;16 为篡改值。"},
    {"claim": "睡眠剥夺 24 小时后,人的操作表现相当于血液酒精浓度 0.12% 的水平。",
     "label": "contradicted", "category": "number_alter", "hops": 1,
     "evidence_docs": ["sleep_zh.md"],
     "rationale": "源文 0.10%;0.12 为篡改值。"},
    {"claim": "咖啡一爆通常发生在豆温 196 至 208 摄氏度之间。",
     "label": "contradicted", "category": "number_alter", "hops": 1,
     "evidence_docs": ["coffee_en.md"],
     "rationale": "源文 196 至 205;208 为篡改值。"},
    {"claim": "哥本哈根通勤者中骑车上班上学的比例约为 59%。",
     "label": "contradicted", "category": "number_alter", "hops": 1,
     "evidence_docs": ["cycling_zh.md"],
     "rationale": "源文 49%;59 为篡改值。"},
    {"claim": "世界卫生组织建议城市居民步行 800 米内应能到达公共绿地。",
     "label": "contradicted", "category": "number_alter", "hops": 1,
     "evidence_docs": ["greenspace_zh.md"],
     "rationale": "源文 300 米;800 为篡改值。"},

    # ------------------------------------------------------------------
    # contradicted / direction
    # ------------------------------------------------------------------
    {"claim": "物理隔离自行车道会使骑行者严重伤害事故上升约 60% 至 75%。",
     "label": "contradicted", "category": "direction", "hops": 1,
     "evidence_docs": ["cycling_zh.md"],
     "rationale": "源文是降低 60% 至 75%,论断把方向反转;数字相同,只能靠方向词判别。"},
    {"claim": "高反射率涂层的浅色路面会使表面温度上升 10 至 20 摄氏度。",
     "label": "contradicted", "category": "direction", "hops": 1,
     "evidence_docs": ["greenspace_zh.md"],
     "rationale": "源文是降低 10 至 20 摄氏度。"},
    {"claim": "遮阴栽培会降低茶氨酸含量并让茶叶更苦涩。",
     "label": "contradicted", "category": "direction", "hops": 1,
     "evidence_docs": ["tea_zh.md"],
     "rationale": "源文: 遮阴可显著提高茶氨酸含量、降低苦涩味,论断把两个方向都反了。"
                  "无数字,词法引擎很难发现。"},
    {"claim": "睡前使用发光电子设备会促进褪黑素分泌。",
     "label": "contradicted", "category": "direction", "hops": 1,
     "evidence_docs": ["sleep_zh.md"],
     "rationale": "源文: 会抑制褪黑素分泌。无数字的方向错误。"},

    # ------------------------------------------------------------------
    # contradicted / entity_swap
    # ------------------------------------------------------------------
    {"claim": "阿姆斯特丹自 2005 年起累计投资超过 20 亿丹麦克朗建设自行车基础设施。",
     "label": "contradicted", "category": "entity_swap", "hops": 1,
     "evidence_docs": ["cycling_zh.md"],
     "rationale": "源文说的是哥本哈根;事实被安到阿姆斯特丹头上。"},
    {"claim": "哥本哈根的自行车出行占全市交通总量的比例约为 38%。",
     "label": "contradicted", "category": "entity_swap", "hops": 1,
     "evidence_docs": ["cycling_zh.md"],
     "rationale": "38% 是阿姆斯特丹的数字,哥本哈根是 49% 的通勤比例。"},
    {"claim": "挪威的自行车盗窃率约为每年每千辆 15 辆。",
     "label": "contradicted", "category": "entity_swap", "hops": 1,
     "evidence_docs": ["cycling_zh.md"],
     "rationale": "源文是荷兰的盗窃率;国家被换掉。"},

    # ------------------------------------------------------------------
    # contradicted / unit
    # ------------------------------------------------------------------
    {"claim": "浓缩咖啡的标准萃取压力为 15 bar。",
     "label": "contradicted", "category": "unit", "hops": 1,
     "evidence_docs": ["coffee_en.md"], "rationale": "源文 9 bar。"},
    {"claim": "一爆之后,浅色烘焙通常在豆温 230 至 240 摄氏度出豆。",
     "label": "contradicted", "category": "unit", "hops": 1,
     "evidence_docs": ["coffee_en.md"],
     "rationale": "源文浅烘约 205 至 215;224 至 230 是二爆(深烘)。"},

    # ------------------------------------------------------------------
    # unsupported / plausible_absent — sounds reasonable, corpus silent
    # ------------------------------------------------------------------
    {"claim": "阿姆斯特丹的自行车盗窃案在 2024 年下降了 12%。",
     "label": "unsupported", "category": "plausible_absent", "hops": 1,
     "evidence_docs": [],
     "rationale": "源文只给盗窃率 15‰ 与 30% 找回率,没有 2024 年降幅。"},
    {"claim": "新加坡要求新建住宅项目在屋顶安装太阳能板。",
     "label": "unsupported", "category": "plausible_absent", "hops": 1,
     "evidence_docs": [],
     "rationale": "源文只提到新加坡的立面绿化与天空花园补偿制度,没有太阳能板。"},
    {"claim": "中国在 2024 年发布了城市绿地建设的新国家标准。",
     "label": "unsupported", "category": "plausible_absent", "hops": 1,
     "evidence_docs": [],
     "rationale": "语料没有任何中国绿地标准的记载。"},
    {"claim": "咖啡烘焙企业普遍采用人工智能来控制烘焙曲线。",
     "label": "unsupported", "category": "plausible_absent", "hops": 1,
     "evidence_docs": [],
     "rationale": "语料完全没有 AI 控制烘焙的内容。"},
    {"claim": "蜜蜂的蜂群数量在 2023 年全球范围内增长了 8%。",
     "label": "unsupported", "category": "plausible_absent", "hops": 1,
     "evidence_docs": [],
     "rationale": "语料只谈美国受管理蜂群数量增长,没有 2023 年全球 8% 的数据。"},
    {"claim": "CBT-I 通常由精神科医生在医院内以团体形式开展。",
     "label": "unsupported", "category": "plausible_absent", "hops": 1,
     "evidence_docs": [],
     "rationale": "语料只讲 CBT-I 的技术与疗程,未提实施形式与人员。"},

    # ------------------------------------------------------------------
    # unsupported / near_miss — adjacent subtopic covered, this fact not
    # ------------------------------------------------------------------
    {"claim": "日本的玉露茶产区主要分布在静冈县和福冈县。",
     "label": "unsupported", "category": "near_miss", "hops": 1,
     "evidence_docs": [],
     "rationale": "语料提到日本玉露茶的遮阴做法,但没有任何产区信息。"},
    {"claim": "荷兰的自行车头盔强制立法始于 2010 年。",
     "label": "unsupported", "category": "near_miss", "hops": 1,
     "evidence_docs": [],
     "rationale": "语料的头盔立法讲的是澳大利亚 1990 年代,不是荷兰 2010 年。"},
    {"claim": "研究表明,城市绿地面积每增加 10%,周边房价上涨约 3%。",
     "label": "unsupported", "category": "near_miss", "hops": 1,
     "evidence_docs": [],
     "rationale": "语料有绿地降温与冷岛范围,完全没有房价或溢出效应研究。"},
    {"claim": "Agtron 数值越大表示烘焙程度越深。",
     "label": "contradicted", "category": "direction", "hops": 1,
     "evidence_docs": ["coffee_en.md"],
     "rationale": "源文: higher numbers indicate lighter roasts,方向正好相反。"},
]


def _check_cases() -> list[str]:
    problems: list[str] = []
    corpus = "\n".join(p.read_text(encoding="utf-8") for p in sorted(SOURCES.glob("*.md")))
    corpus_nums = distinctive_numbers(corpus)
    for i, c in enumerate(CASES, 1):
        if c["label"] not in ("supported", "unsupported", "contradicted"):
            problems.append(f"case {i}: bad label {c['label']!r}")
        for doc in c.get("evidence_docs", []):
            if not (SOURCES / doc).exists():
                problems.append(f"case {i}: missing doc {doc}")
        if c["label"] == "supported" and not c.get("evidence_docs"):
            problems.append(f"case {i}: supported without evidence_docs")
        if c["hops"] > 1 and len(c.get("evidence_docs", [])) < 1:
            problems.append(f"case {i}: multi-hop without evidence_docs")
        if c["category"] == "number_alter":
            nums = distinctive_numbers(c["claim"])
            supported_nums = {n for n in nums if n not in corpus_nums}
            if not supported_nums:
                problems.append(f"case {i}: number_alter has no corpus-absent number")
    return problems


def main() -> int:
    problems = _check_cases()
    if problems:
        for p in problems:
            print(f"[hard-gold] SELF-CHECK FAIL: {p}", file=sys.stderr)
        return 1

    records = [dict(c) for c in CASES]
    rng = random.Random(SEED)
    rng.shuffle(records)
    for i, r in enumerate(records):
        r["id"] = f"H{i + 1:04d}"
    order = ("id", "claim", "label", "category", "hops", "evidence_docs",
             "rationale")
    lines = [json.dumps({k: r[k] for k in order}, ensure_ascii=False)
             for r in records]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    dist = Counter(r["label"] for r in records)
    cats = Counter(r["category"] for r in records)
    print(f"hard gold -> {OUT}: {len(records)} claims {dict(dist)}")
    print(f"  categories: {dict(sorted(cats.items()))}")
    print(f"  multi-hop : {sum(1 for r in records if r['hops'] > 1)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
