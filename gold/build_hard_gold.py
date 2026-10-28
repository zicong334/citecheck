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

    # ==================================================================
    # v0.3 扩充:49 -> 150。以下每条都对齐 gold/sources 语料,固定 seed 幂等重建。
    # 新增类别:magnitude(量级错误)、temporal(时间/时态错配)、modal(情态篡改)。
    # ==================================================================

    # ---------------- supported / verbatim(近原文,回归锚点)----------------
    {"claim": "First crack typically occurs when bean temperature reaches 196 to "
              "205 degrees Celsius.",
     "label": "supported", "category": "verbatim", "hops": 1,
     "evidence_docs": ["coffee_en.md"], "rationale": "coffee_en 第 2 段原文。"},
    {"claim": "A queen bee can lay up to 2,000 eggs per day during the spring peak.",
     "label": "supported", "category": "verbatim", "hops": 1,
     "evidence_docs": ["bees_en.md"], "rationale": "bees_en 第 1 段原文。"},
    {"claim": "茶树适宜生长的年均温度为 15 至 25 摄氏度,年降水量需要达到 1000 毫米以上。",
     "label": "supported", "category": "verbatim", "hops": 1,
     "evidence_docs": ["tea_zh.md"], "rationale": "tea_zh 第 1 段原文。"},
    {"claim": "整夜 REM 睡眠约占总睡眠时间的 20% 至 25%。",
     "label": "supported", "category": "verbatim", "hops": 1,
     "evidence_docs": ["sleep_zh.md"], "rationale": "sleep_zh 第 1 段原文。"},
    {"claim": "世界卫生组织建议城市居民步行 300 米内应能到达公共绿地。",
     "label": "supported", "category": "verbatim", "hops": 1,
     "evidence_docs": ["greenspace_zh.md"], "rationale": "greenspace_zh 第 3 段原文。"},
    {"claim": "物理隔离自行车道可将骑行者严重伤害事故降低约 60% 至 75%。",
     "label": "supported", "category": "verbatim", "hops": 1,
     "evidence_docs": ["cycling_zh.md"], "rationale": "cycling_zh 第 3 段原文。"},

    # ---------------- supported / paraphrase_zh(中文同义改写)----------------
    {"claim": "制作红茶时,儿茶素在多酚氧化酶的催化下转变成茶黄素和茶红素。",
     "label": "supported", "category": "paraphrase_zh", "hops": 1,
     "evidence_docs": ["tea_zh.md"],
     "rationale": "源:红茶发酵儿茶素在多酚氧化酶作用下生成茶黄素和茶红素;同义改写。"},
    {"claim": "给茶树遮阴能显著提高茶叶中的茶氨酸,并减轻苦涩味。",
     "label": "supported", "category": "paraphrase_zh", "hops": 1,
     "evidence_docs": ["tea_zh.md"],
     "rationale": "源:遮阴可显著提高茶氨酸含量、降低苦涩味;方向一致。"},
    {"claim": "连续两周每晚只睡六小时的人,认知表现受损程度和连续两晚完全被剥夺睡眠差不多。",
     "label": "supported", "category": "paraphrase_zh", "hops": 1,
     "evidence_docs": ["sleep_zh.md"],
     "rationale": "源:连续两周每晚 6 小时的认知下降与连续两晚完全剥夺相当。"},
    {"claim": "夏季屋顶绿化可使顶层室内温度降低 2 至 4 摄氏度,并削减屋面径流一半以上。",
     "label": "supported", "category": "paraphrase_zh", "hops": 1,
     "evidence_docs": ["greenspace_zh.md"],
     "rationale": "源:屋顶绿化使顶层降温 2 至 4 摄氏度,削减径流 50% 以上。"},
    {"claim": "在哥本哈根,差不多每两个通勤者里就有一个骑车上班上学。",
     "label": "supported", "category": "paraphrase_zh", "hops": 1,
     "evidence_docs": ["cycling_zh.md"],
     "rationale": "源:哥本哈根通勤者骑车比例约 49%(≈一半);无数字改写。"},
    {"claim": "褪黑素由松果体在黑暗中分泌,通常在习惯性就寝时间前两三个小时开始升高。",
     "label": "supported", "category": "paraphrase_zh", "hops": 1,
     "evidence_docs": ["sleep_zh.md"],
     "rationale": "源:褪黑素由松果体黑暗中分泌,habitual bedtime 前 2 至 3 小时升高。"},
    {"claim": "绿茶的加工要经过杀青、揉捻和干燥这几道主要工序。",
     "label": "supported", "category": "paraphrase_zh", "hops": 1,
     "evidence_docs": ["tea_zh.md"], "rationale": "源:绿茶经过杀青、揉捻、干燥三道主要工序。"},
    {"claim": "失眠的认知行为疗法中,睡眠限制在刚开始的阶段会加重白天的困倦。",
     "label": "supported", "category": "paraphrase_zh", "hops": 1,
     "evidence_docs": ["sleep_zh.md"], "rationale": "源:睡眠限制疗法初期会加重日间困倦。"},
    {"claim": "氮肥施用过量会让茶叶里的咖啡碱含量升高,同时茶氨酸含量下降。",
     "label": "supported", "category": "paraphrase_zh", "hops": 1,
     "evidence_docs": ["tea_zh.md"], "rationale": "源:氮肥过量导致咖啡碱升高、茶氨酸下降。"},
    {"claim": "城市中成片绿地的冷岛效应,可以延伸到周边一百至五百米的范围。",
     "label": "supported", "category": "paraphrase_zh", "hops": 1,
     "evidence_docs": ["greenspace_zh.md"], "rationale": "源:成片绿地冷岛效应延伸周边 100 至 500 米。"},

    # ---------------- supported / crosslingual(跨语言,多数无阿拉伯数字)----------------
    {"claim": "蜜蜂跳摇摆舞时,舞蹈相对竖直方向的角度编码了食物相对太阳的方位。",
     "label": "supported", "category": "crosslingual", "hops": 1,
     "evidence_docs": ["bees_en.md"],
     "rationale": "bees_en: angle of waggle run relative to vertical encodes direction "
                  "relative to the sun;中文论断、英文源、无数字。"},
    {"claim": "咖啡烘焙中的美拉德反应会生成数百种芳香化合物。",
     "label": "supported", "category": "crosslingual", "hops": 1,
     "evidence_docs": ["coffee_en.md"],
     "rationale": "coffee_en: Maillard reaction produces several hundred aromatic compounds。"},
    {"claim": "萃取不足的浓缩咖啡会偏酸,因为有机酸溶解得比较早。",
     "label": "supported", "category": "crosslingual", "hops": 1,
     "evidence_docs": ["coffee_en.md"],
     "rationale": "coffee_en: under-extracted espresso tastes sour because acids dissolve early。"},
    {"claim": "一个蜂群要拜访大约十万朵花,才能产出一公斤蜂蜜。",
     "label": "supported", "category": "crosslingual", "hops": 1,
     "evidence_docs": ["bees_en.md"],
     "rationale": "bees_en: a colony may visit around 100,000 flowers per kilogram of honey。"},
    {"claim": "同一种豆子,浅烘与深烘之间的咖啡因差异按重量算通常不超过百分之五。",
     "label": "supported", "category": "crosslingual", "hops": 1,
     "evidence_docs": ["coffee_en.md"],
     "rationale": "coffee_en: difference between light and dark roast only about 5 percent "
                  "or less by mass。"},
    {"claim": "Green tea is processed with high-temperature fixation to deactivate "
              "polyphenol oxidase.",
     "label": "supported", "category": "crosslingual", "hops": 1,
     "evidence_docs": ["tea_zh.md"],
     "rationale": "tea_zh: 绿茶高温杀青钝化多酚氧化酶;英文论断、中文源、无数字。"},
    {"claim": "A mature tree's daily transpiration has a cooling effect comparable to "
              "running two or three household air conditioners.",
     "label": "supported", "category": "crosslingual", "hops": 1,
     "evidence_docs": ["greenspace_zh.md"],
     "rationale": "greenspace_zh: 乔木蒸腾日均 200 至 400 升,相当于 2 至 3 台家用空调。"},
    {"claim": "Chronic sleep deprivation makes people underestimate their own impairment.",
     "label": "supported", "category": "crosslingual", "hops": 1,
     "evidence_docs": ["sleep_zh.md"],
     "rationale": "sleep_zh: 慢性睡眠不足者会低估自己的受损程度;英文论断、中文源。"},
    {"claim": "Copenhagen built a green-wave signal system so that cyclists can keep "
              "passing intersections without hitting red lights.",
     "label": "supported", "category": "crosslingual", "hops": 1,
     "evidence_docs": ["cycling_zh.md"],
     "rationale": "cycling_zh: 哥本哈根绿波信号系统,骑行者连续通过路口不遇红灯。"},
    {"claim": "咖啡行业用 Agtron 值量化烘焙颜色,数值越高表示烘焙越浅。",
     "label": "supported", "category": "crosslingual", "hops": 1,
     "evidence_docs": ["coffee_en.md"],
     "rationale": "coffee_en: higher Agtron numbers indicate lighter roasts;此论断与源一致"
                  "(与 direction 类的反转用例互为对照)。"},

    # ---------------- supported / multi_hop(跨段落两段证据)----------------
    {"claim": "茶多酚占干重的 18% 至 36%,而茶黄素一般占红茶干重的 0.5% 至 2%。",
     "label": "supported", "category": "multi_hop", "hops": 2,
     "evidence_docs": ["tea_zh.md"], "rationale": "第 2 段茶多酚比例 + 第 3 段茶黄素比例。"},
    {"claim": "Green beans start at about 8 to 12 percent moisture, and first crack "
              "happens at 196 to 205 degrees Celsius.",
     "label": "supported", "category": "multi_hop", "hops": 2,
     "evidence_docs": ["coffee_en.md"], "rationale": "第 1 段含水率 + 第 2 段一爆温度。"},
    {"claim": "Colonies hold 20,000 to 80,000 bees, and a single forager carries about "
              "40 to 70 milligrams of nectar per trip.",
     "label": "supported", "category": "multi_hop", "hops": 2,
     "evidence_docs": ["bees_en.md"], "rationale": "第 1 段蜂群规模 + 第 3 段采集蜂载蜜量。"},
    {"claim": "阿姆斯特丹的自行车出行约占全市交通总量的 38%,而哥本哈根自 2005 年起"
              "累计投资超过 20 亿丹麦克朗。",
     "label": "supported", "category": "multi_hop", "hops": 2,
     "evidence_docs": ["cycling_zh.md"], "rationale": "第 1 段阿姆斯特丹 + 第 2 段哥本哈根投资。"},
    {"claim": "树冠遮阴可使地表温度降低 10 至 25 摄氏度,而高反射率的浅色路面可使表面"
              "温度降低 10 至 20 摄氏度。",
     "label": "supported", "category": "multi_hop", "hops": 2,
     "evidence_docs": ["greenspace_zh.md"], "rationale": "第 2 段树冠降温 + 第 5 段反射路面降温。"},
    {"claim": "整夜 REM 睡眠约占总睡眠的 20% 至 25%,而睡前阅读电子书 4 小时会让褪黑素"
              "分泌减少约 50%。",
     "label": "supported", "category": "multi_hop", "hops": 2,
     "evidence_docs": ["sleep_zh.md"], "rationale": "第 1 段 REM 占比 + 第 3 段电子书影响。"},
    {"claim": "全球天然蜂蜜产量在 2022 年约为 190 万吨,而加利福尼亚的扁桃园每年二月"
              "需要约 180 万个蜂群。",
     "label": "supported", "category": "multi_hop", "hops": 2,
     "evidence_docs": ["bees_en.md"], "rationale": "第 3 段蜂蜜产量 + 第 5 段扁桃授粉蜂群。"},

    # ---------------- contradicted / number_alter(篡改特征数字)----------------
    {"claim": "茶树适宜生长的年均温度为 15 至 35 摄氏度。",
     "label": "contradicted", "category": "number_alter", "hops": 1,
     "evidence_docs": ["tea_zh.md"], "rationale": "源 15 至 25;35 为篡改值。"},
    {"claim": "2023 年中国茶叶产量约为 375 万吨,位居世界第一。",
     "label": "contradicted", "category": "number_alter", "hops": 1,
     "evidence_docs": ["tea_zh.md"], "rationale": "源 355 万吨;375 为篡改值。"},
    {"claim": "一爆通常发生在豆温 196 至 218 摄氏度之间。",
     "label": "contradicted", "category": "number_alter", "hops": 1,
     "evidence_docs": ["coffee_en.md"], "rationale": "源 196 至 205;218 为篡改值。"},
    {"claim": "浓缩咖啡通常在 9 bar 压力下萃取,水温保持在 90 至 106 摄氏度。",
     "label": "contradicted", "category": "number_alter", "hops": 1,
     "evidence_docs": ["coffee_en.md"], "rationale": "源水温 90 至 96;106 为篡改值。"},
    {"claim": "阿姆斯特丹的自行车出行约占全市交通总量的 43%。",
     "label": "contradicted", "category": "number_alter", "hops": 1,
     "evidence_docs": ["cycling_zh.md"], "rationale": "源 38%;43 为篡改值。"},
    {"claim": "北京回龙观至上地的自行车专用路全长约 8.5 公里。",
     "label": "contradicted", "category": "number_alter", "hops": 1,
     "evidence_docs": ["cycling_zh.md"], "rationale": "源 6.5 公里;8.5 为篡改值。"},
    {"claim": "树冠遮阴可使地表温度降低 10 至 28 摄氏度。",
     "label": "contradicted", "category": "number_alter", "hops": 1,
     "evidence_docs": ["greenspace_zh.md"], "rationale": "源 10 至 25;28 为篡改值。"},
    {"claim": "睡眠剥夺 24 小时后,操作表现相当于血液酒精浓度 0.14% 的水平。",
     "label": "contradicted", "category": "number_alter", "hops": 1,
     "evidence_docs": ["sleep_zh.md"], "rationale": "源 0.10%;0.14 为篡改值。"},
    {"claim": "A queen bee can lay up to 3,000 eggs per day during the spring peak.",
     "label": "contradicted", "category": "number_alter", "hops": 1,
     "evidence_docs": ["bees_en.md"], "rationale": "源 2,000;3,000 为篡改值。"},
    {"claim": "全球天然蜂蜜产量在 2022 年约为 210 万吨。",
     "label": "contradicted", "category": "number_alter", "hops": 1,
     "evidence_docs": ["bees_en.md"], "rationale": "源约 190 万吨(1.9 million);210 为篡改值。"},

    # ---------------- contradicted / direction(方向反转)----------------
    {"claim": "树冠遮阴会使地表温度升高 10 至 25 摄氏度。",
     "label": "contradicted", "category": "direction", "hops": 1,
     "evidence_docs": ["greenspace_zh.md"], "rationale": "源为降低 10 至 25;方向反转。"},
    {"claim": "睡前阅读电子书 4 小时会使褪黑素分泌增加约 50%。",
     "label": "contradicted", "category": "direction", "hops": 1,
     "evidence_docs": ["sleep_zh.md"], "rationale": "源为减少约 50%;方向反转(数字相同)。"},
    {"claim": "洛杉矶试用反射涂层后,路面温度平均上升约 5 至 10 摄氏度。",
     "label": "contradicted", "category": "direction", "hops": 1,
     "evidence_docs": ["greenspace_zh.md"], "rationale": "源为平均下降约 5 至 10;方向反转。"},
    {"claim": "夏季屋顶绿化会使顶层室内温度上升 2 至 4 摄氏度。",
     "label": "contradicted", "category": "direction", "hops": 1,
     "evidence_docs": ["greenspace_zh.md"], "rationale": "源为降低 2 至 4;方向反转。"},
    {"claim": "睡眠限制疗法会降低睡眠效率。",
     "label": "contradicted", "category": "direction", "hops": 1,
     "evidence_docs": ["sleep_zh.md"], "rationale": "源为数周后睡眠效率提升至 90% 左右;方向反转。"},
    {"claim": "咖啡豆烘焙得越深,Agtron 数值就越高。",
     "label": "contradicted", "category": "direction", "hops": 1,
     "evidence_docs": ["coffee_en.md"], "rationale": "源:higher numbers indicate lighter roasts。"},
    {"claim": "遮阴栽培会降低茶叶中的茶氨酸含量。",
     "label": "contradicted", "category": "direction", "hops": 1,
     "evidence_docs": ["tea_zh.md"], "rationale": "源:遮阴可显著提高茶氨酸含量;方向反转。"},
    {"claim": "城市中心的气温通常低于周边郊区 3 至 5 摄氏度。",
     "label": "contradicted", "category": "direction", "hops": 1,
     "evidence_docs": ["greenspace_zh.md"], "rationale": "源:中心气温高于郊区 3 至 5;方向反转。"},
    {"claim": "修建物理隔离自行车道后,骑行者遭遇严重伤害事故的概率会上升约 60% 至 75%。",
     "label": "contradicted", "category": "direction", "hops": 1,
     "evidence_docs": ["cycling_zh.md"], "rationale": "源为降低约 60% 至 75%;方向反转。"},

    # ---------------- contradicted / entity_swap(张冠李戴)----------------
    {"claim": "巴西的天然蜂蜜产量约为 45 万吨,居各国首位。",
     "label": "contradicted", "category": "entity_swap", "hops": 1,
     "evidence_docs": ["bees_en.md"], "rationale": "源:中国约 450,000 吨居首;实体换成巴西。"},
    {"claim": "中国的咖啡产量约为 6600 万袋,位居世界第一。",
     "label": "contradicted", "category": "entity_swap", "hops": 1,
     "evidence_docs": ["coffee_en.md"], "rationale": "源:巴西约 6600 万袋;实体换成中国。"},
    {"claim": "北京的自行车出行占全市交通总量的比例约为 38%。",
     "label": "contradicted", "category": "entity_swap", "hops": 1,
     "evidence_docs": ["cycling_zh.md"], "rationale": "源:38% 是阿姆斯特丹;实体换成北京。"},
    {"claim": "新加坡自 2017 年起在多条街道试用浅灰色反射涂层路面。",
     "label": "contradicted", "category": "entity_swap", "hops": 1,
     "evidence_docs": ["greenspace_zh.md"], "rationale": "源:洛杉矶 2017 年试用;实体换成新加坡。"},
    {"claim": "越南的扁桃园每年二月需要约 180 万个蜂群授粉。",
     "label": "contradicted", "category": "entity_swap", "hops": 1,
     "evidence_docs": ["bees_en.md"], "rationale": "源:加利福尼亚扁桃园;实体换成越南。"},
    {"claim": "丹麦的自行车出行占全市交通总量的比例约为 38%。",
     "label": "contradicted", "category": "entity_swap", "hops": 1,
     "evidence_docs": ["cycling_zh.md"], "rationale": "源:38% 是阿姆斯特丹;实体换成丹麦。"},
    {"claim": "洛杉矶的城市绿化覆盖率维持在约 50%。",
     "label": "contradicted", "category": "entity_swap", "hops": 1,
     "evidence_docs": ["greenspace_zh.md"], "rationale": "源:新加坡绿化覆盖率约 50%;实体换成洛杉矶。"},
    {"claim": "日本的茶叶产量在 2023 年约为 355 万吨,位居世界第一。",
     "label": "contradicted", "category": "entity_swap", "hops": 1,
     "evidence_docs": ["tea_zh.md"], "rationale": "源:中国 355 万吨世界第一;实体换成日本。"},
    {"claim": "阿姆斯特丹已规划超过 750 公里的自行车高速路网。",
     "label": "contradicted", "category": "entity_swap", "hops": 1,
     "evidence_docs": ["cycling_zh.md"], "rationale": "源:哥本哈根都市区 750 公里;实体换成阿姆斯特丹。"},

    # ---------------- contradicted / unit(单位/量纲错误)----------------
    {"claim": "浓缩咖啡的标准萃取压力为 19 bar。",
     "label": "contradicted", "category": "unit", "hops": 1,
     "evidence_docs": ["coffee_en.md"], "rationale": "源 9 bar;19 为错误取值。"},
    {"claim": "佛得草式轻型屋顶绿化的荷载低于每平方米 1500 公斤。",
     "label": "contradicted", "category": "unit", "hops": 1,
     "evidence_docs": ["greenspace_zh.md"], "rationale": "源 150 公斤/平米;1500 为错误取值。"},
    {"claim": "成人每个睡眠周期约持续 900 分钟。",
     "label": "contradicted", "category": "unit", "hops": 1,
     "evidence_docs": ["sleep_zh.md"], "rationale": "源约 90 分钟;900 为错误取值。"},
    {"claim": "单只采集蜂每次携带约 400 至 700 毫克花蜜。",
     "label": "contradicted", "category": "unit", "hops": 1,
     "evidence_docs": ["bees_en.md"], "rationale": "源 40 至 70 毫克;放大十倍。"},
    {"claim": "北京回龙观至上地的自行车专用路全长约 13 公里。",
     "label": "contradicted", "category": "unit", "hops": 1,
     "evidence_docs": ["cycling_zh.md"], "rationale": "源约 6.5 公里;13 为错误取值。"},
    {"claim": "绿茶杀青温度通常控制在 250 摄氏度以上。",
     "label": "contradicted", "category": "unit", "hops": 1,
     "evidence_docs": ["tea_zh.md"], "rationale": "源 200 摄氏度以上;250 为错误取值。"},

    # ---------------- contradicted / magnitude(量级错误:十倍/百倍)NEW ----------------
    {"claim": "一个蜂群要拜访约 100 万朵花,才能产出一公斤蜂蜜。",
     "label": "contradicted", "category": "magnitude", "hops": 1,
     "evidence_docs": ["bees_en.md"], "rationale": "源约 10 万朵(100,000);放大十倍。"},
    {"claim": "全球天然蜂蜜产量在 2022 年约为 1900 万吨。",
     "label": "contradicted", "category": "magnitude", "hops": 1,
     "evidence_docs": ["bees_en.md"], "rationale": "源约 190 万吨(1.9 million);放大十倍。"},
    {"claim": "2023 年中国茶叶产量约为 3550 万吨。",
     "label": "contradicted", "category": "magnitude", "hops": 1,
     "evidence_docs": ["tea_zh.md"], "rationale": "源约 355 万吨;放大十倍。"},
    {"claim": "一个蜂群在温带地区通常有 20 万至 80 万只个体。",
     "label": "contradicted", "category": "magnitude", "hops": 1,
     "evidence_docs": ["bees_en.md"], "rationale": "源 2 万至 8 万(20,000-80,000);放大十倍。"},
    {"claim": "阿姆斯特丹拥有超过 800 万辆自行车。",
     "label": "contradicted", "category": "magnitude", "hops": 1,
     "evidence_docs": ["cycling_zh.md"], "rationale": "源超过 80 万辆;放大十倍。"},
    {"claim": "巴西的咖啡产量约为 6.6 亿袋。",
     "label": "contradicted", "category": "magnitude", "hops": 1,
     "evidence_docs": ["coffee_en.md"], "rationale": "源约 6600 万袋(66 million);放大十倍。"},
    {"claim": "连续 24 小时不睡相当于血液酒精浓度 1.0% 的水平。",
     "label": "contradicted", "category": "magnitude", "hops": 1,
     "evidence_docs": ["sleep_zh.md"], "rationale": "源 0.10%;放大十倍。"},
    {"claim": "咖啡生豆的含水率高达 80% 至 120%。",
     "label": "contradicted", "category": "magnitude", "hops": 1,
     "evidence_docs": ["coffee_en.md"], "rationale": "源约 8% 至 12%;放大十倍。"},

    # ---------------- contradicted / temporal(时间/时态错配)NEW ----------------
    {"claim": "北京回龙观至上地的自行车专用路将于 2029 年 5 月才开通。",
     "label": "contradicted", "category": "temporal", "hops": 1,
     "evidence_docs": ["cycling_zh.md"], "rationale": "源:2019 年 5 月已开通;时态与年份均错。"},
    {"claim": "阿姆斯特丹中央车站前的水下自行车停车场尚未建成,预计 2033 年启用。",
     "label": "contradicted", "category": "temporal", "hops": 1,
     "evidence_docs": ["cycling_zh.md"], "rationale": "源:2023 年已启用;时态与年份均错。"},
    {"claim": "蜂群崩溃失调从 2016 年前后开始受到全球关注。",
     "label": "contradicted", "category": "temporal", "hops": 1,
     "evidence_docs": ["bees_en.md"], "rationale": "源:from 2006 onward;年份错配。"},
    {"claim": "全球咖啡产量在 2003/04 季约为 1.78 亿个 60 公斤袋。",
     "label": "contradicted", "category": "temporal", "hops": 1,
     "evidence_docs": ["coffee_en.md"], "rationale": "源:2023/24 季;年份错配。"},
    {"claim": "中国多地中小学自 2011 年起推行睡眠管理规定。",
     "label": "contradicted", "category": "temporal", "hops": 1,
     "evidence_docs": ["sleep_zh.md"], "rationale": "源:自 2021 年起;年份错配。"},
    {"claim": "洛杉矶自 2007 年起在多条街道试用浅灰色反射涂层路面。",
     "label": "contradicted", "category": "temporal", "hops": 1,
     "evidence_docs": ["greenspace_zh.md"], "rationale": "源:自 2017 年起;年份错配。"},

    # ---------------- contradicted / modal(情态篡改:可能/存疑→确定/唯一)NEW ----------------
    {"claim": "科学家已确定瓦螨是蜜蜂越冬死亡的唯一原因。",
     "label": "contradicted", "category": "modal", "hops": 1,
     "evidence_docs": ["bees_en.md"],
     "rationale": "源:归因于瓦螨+病毒+农药+营养等多因素的组合,而非单一原因。"},
    {"claim": "强制头盔立法对骑行安全的净效益在学界已有定论。",
     "label": "contradicted", "category": "modal", "hops": 1,
     "evidence_docs": ["cycling_zh.md"], "rationale": "源:学界对强制立法的净效益仍存在讨论。"},
    {"claim": "反射涂层路面已被证实适合大规模推广。",
     "label": "contradicted", "category": "modal", "hops": 1,
     "evidence_docs": ["greenspace_zh.md"], "rationale": "源:学界对大规模推广仍持审慎态度。"},
    {"claim": "美国受管理蜂群数量的增长,说明蜂群崩溃问题已被彻底解决。",
     "label": "contradicted", "category": "modal", "hops": 1,
     "evidence_docs": ["bees_en.md"],
     "rationale": "源:尽管年损失率仍高,靠拆分健康蜂群重建才使数量增长;并非已解决。"},
    {"claim": "遮阴栽培必然能消除茶叶的全部苦涩味。",
     "label": "contradicted", "category": "modal", "hops": 1,
     "evidence_docs": ["tea_zh.md"], "rationale": "源:遮阴可降低苦涩味,并非必然消除全部。"},
    {"claim": "蜜蜂是唯一的动物授粉者。",
     "label": "contradicted", "category": "modal", "hops": 1,
     "evidence_docs": ["bees_en.md"],
     "rationale": "源:约 75% 主要作物受益于动物授粉,蜜蜂只是最受广泛管理的授粉者之一。"},

    # ---------------- unsupported / plausible_absent(语料完全沉默)----------------
    {"claim": "普洱茶的收藏市场价格在过去十年上涨了约三倍。",
     "label": "unsupported", "category": "plausible_absent", "hops": 1,
     "evidence_docs": [], "rationale": "语料讲普洱贮藏条件,从未涉及收藏价格。"},
    {"claim": "冷萃咖啡的咖啡因含量高于热冲咖啡。",
     "label": "unsupported", "category": "plausible_absent", "hops": 1,
     "evidence_docs": [], "rationale": "语料只谈烘焙度与咖啡因,未涉及冷萃。"},
    {"claim": "电动自行车的电池续航普遍超过 100 公里。",
     "label": "unsupported", "category": "plausible_absent", "hops": 1,
     "evidence_docs": [], "rationale": "语料只提到专用路禁电动自行车,未谈续航。"},
    {"claim": "垂直花园的自动灌溉系统通常由太阳能供电。",
     "label": "unsupported", "category": "plausible_absent", "hops": 1,
     "evidence_docs": [], "rationale": "语料提到立面绿化/天空花园,未涉及灌溉供电。"},
    {"claim": "午睡超过 30 分钟会显著增加心血管疾病风险。",
     "label": "unsupported", "category": "plausible_absent", "hops": 1,
     "evidence_docs": [], "rationale": "语料未讨论午睡与心血管风险。"},
    {"claim": "有机农业能完全避免农药对蜜蜂的伤害。",
     "label": "unsupported", "category": "plausible_absent", "hops": 1,
     "evidence_docs": [], "rationale": "语料把农药列为损失因素之一,未谈有机农业。"},

    # ---------------- unsupported / near_miss(相邻话题,该事实未给)----------------
    {"claim": "阿姆斯特丹的自行车高速路网总长已超过 500 公里。",
     "label": "unsupported", "category": "near_miss", "hops": 1,
     "evidence_docs": [], "rationale": "语料的 750 公里高速路网属哥本哈根,未给阿姆斯特丹的数据。"},
    {"claim": "新加坡的屋顶绿化覆盖率在全球城市中排名第一。",
     "label": "unsupported", "category": "near_miss", "hops": 1,
     "evidence_docs": [], "rationale": "语料讲新加坡绿化覆盖率约 50%,未做全球排名。"},
    {"claim": "褪黑素补充剂能安全有效地改善成人失眠。",
     "label": "unsupported", "category": "near_miss", "hops": 1,
     "evidence_docs": [], "rationale": "语料讲褪黑素分泌与 CBT-I,未涉及补充剂疗效。"},
    {"claim": "铁观音属于乌龙茶,主产于福建安溪。",
     "label": "unsupported", "category": "near_miss", "hops": 1,
     "evidence_docs": [], "rationale": "语料提到乌龙茶半发酵、福建是主产区之一,但没有铁观音/安溪。"},
    {"claim": "透水铺装的使用寿命通常比沥青路面长一倍。",
     "label": "unsupported", "category": "near_miss", "hops": 1,
     "evidence_docs": [], "rationale": "语料讲透水铺装降温,未比较其与沥青的寿命。"},
    {"claim": "共享单车的维护成本主要由用户押金覆盖。",
     "label": "unsupported", "category": "near_miss", "hops": 1,
     "evidence_docs": [], "rationale": "语料只给共享单车平均寿命,未涉及维护成本或押金。"},
    {"claim": "玉露茶在采收前覆盖的遮阴网通常为黑色尼龙材质。",
     "label": "unsupported", "category": "near_miss", "hops": 1,
     "evidence_docs": [], "rationale": "语料给遮阴网遮光率 90%,未说明材质颜色。"},
    {"claim": "咖啡生豆在烘焙前的含水率会随产地海拔显著变化。",
     "label": "unsupported", "category": "near_miss", "hops": 1,
     "evidence_docs": [], "rationale": "语料给生豆含水率 8% 至 12%,未讨论海拔对其影响。"},
]


def _check_cases() -> list[str]:
    problems: list[str] = []
    corpus = "\n".join(p.read_text(encoding="utf-8") for p in sorted(SOURCES.glob("*.md")))
    corpus_nums = distinctive_numbers(corpus)
    # 这些类别靠"改出一个语料里没有的数值"成立,必须验证篡改值确实不在语料中。
    # unit 不在此列:单位错误的取值(如 15 bar 的 15)可能在语料别处巧合出现,
    # 但在该事实槽里仍是错值,由判官的同槽竞争取值逻辑负责识别。
    altered_cats = {"number_alter", "magnitude", "temporal"}
    seen_claims: dict[str, int] = {}
    allowed_cats = {
        "verbatim", "paraphrase_zh", "crosslingual", "multi_hop",
        "number_alter", "direction", "entity_swap", "unit", "magnitude",
        "temporal", "modal", "plausible_absent", "near_miss",
    }
    for i, c in enumerate(CASES, 1):
        if c["label"] not in ("supported", "unsupported", "contradicted"):
            problems.append(f"case {i}: bad label {c['label']!r}")
        if not c.get("rationale"):
            problems.append(f"case {i}: 缺少 rationale")
        if c["category"] not in allowed_cats:
            problems.append(f"case {i}: 未知类别 {c['category']!r}")
        for doc in c.get("evidence_docs", []):
            if not (SOURCES / doc).exists():
                problems.append(f"case {i}: missing doc {doc}")
        if c["label"] == "supported" and not c.get("evidence_docs"):
            problems.append(f"case {i}: supported without evidence_docs")
        if c["hops"] > 1 and len(c.get("evidence_docs", [])) < 1:
            problems.append(f"case {i}: multi-hop without evidence_docs")
        if c["category"] in altered_cats:
            nums = distinctive_numbers(c["claim"])
            absent = {n for n in nums if n not in corpus_nums}
            if not absent:
                problems.append(
                    f"case {i}: {c['category']} 没有篡改出语料外的数字: {sorted(nums)}")
        key = c["claim"].strip()
        if key in seen_claims:
            problems.append(f"case {i}: 与 case {seen_claims[key]} 论断重复")
        seen_claims[key] = i
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
