# -*- coding: utf-8 -*-
"""人工维护的中英概念词表与实体名录。

为什么需要它
------------
v0.1 的检索只有*数字*能跨语言对上:因为数字不随翻译改变
("196 至 205 摄氏度"里仍有 196 和 205)。一条不含数字的跨语言论断,
词法引擎完全看不见,永远召回不到英文证据。

向量通道能通用地解决这个问题,但要么下载模型,要么依赖 API。本模块是
零依赖的中间方案:一张小小的、人工整理的表,把中英两种表面形式映射到
*同一个概念词元*。一对中英表述会归一到同一词元
(如 ``茶树``/``tea`` → ``tea``,``一爆``/``first crack`` → ``first_crack``),
于是 BM25 看到的是真实的词面重合,而不是只靠数字硬撑。

范围与诚实说明
--------------
这是*领域*词表,只覆盖随附语料涉及的六个主题(茶、咖啡、骑行、睡眠、
蜜蜂、城市绿地)。它刻意不做通用翻译词典:开放领域请用
``--retriever hybrid`` 配向量后端(见 :mod:`faithsift.embed`)。
词表外的词会直接落回原始词法词元,所以召回只会平滑退化到 v0.1 的水平,
不会崩掉。
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------
# zh surface form -> concept token(中文表面形式 → 概念词元)
# 按长度从长到短扫描,保证多字词优先于它的组成部分。
# --------------------------------------------------------------------------
ZH: dict[str, str] = {
    # --- tea -------------------------------------------------------------
    "茶树": "tea", "茶叶": "tea", "茶": "tea", "普洱茶": "puerh", "普洱": "puerh",
    "绿茶": "green_tea", "红茶": "black_tea", "乌龙茶": "oolong", "乌龙": "oolong",
    "茶多酚": "polyphenol", "多酚": "polyphenol", "儿茶素": "catechin",
    "咖啡碱": "caffeine", "咖啡因": "caffeine", "茶氨酸": "theanine",
    "杀青": "fixation", "揉捻": "rolling", "干燥": "drying",
    "发酵": "fermentation", "半发酵": "semi_fermented", "全发酵": "full_fermented",
    "茶黄素": "theaflavin", "茶红素": "thearubigin",
    "主产区": "producing_region", "产区": "producing_region",
    "产量": "output", "氮肥": "nitrogen_fertilizer", "氮": "nitrogen",
    "酸性土": "acidic_soil", "土壤": "soil", "遮阴": "shade", "遮光": "shade",
    "相对湿度": "humidity", "贮藏": "storage", "储存": "storage",
    "钝化": "inactivation", "酶": "enzyme", "多酚氧化酶": "ppo",
    "香气": "aroma", "香味": "aroma", "褐色": "brown", "棕色": "brown",
    # --- coffee ----------------------------------------------------------
    "咖啡": "coffee", "烘焙": "roast", "烘烤": "roast", "生豆": "green_bean",
    "绿咖啡": "green_bean", "一爆": "first_crack", "二爆": "second_crack",
    "浅烘": "light_roast", "深烘": "dark_roast", "浅焙": "light_roast",
    "深焙": "dark_roast", "深度烘焙": "dark_roast", "浅度烘焙": "light_roast",
    "美拉德": "maillard", "绿原酸": "chlorogenic",
    "抗氧化": "antioxidant", "艾格壮": "agtron", "浓缩咖啡": "espresso",
    "意式浓缩": "espresso", "萃取": "extraction", "压力": "pressure",
    "含水率": "moisture", "水分": "moisture", "全球产量": "output",
    # --- cycling ---------------------------------------------------------
    "自行车": "bicycle", "单车": "bicycle", "骑行": "cycling", "骑行者": "cyclist",
    "自行车道": "bike_lane", "专用路": "bike_lane", "隔离": "separation",
    "物理隔离": "separation", "绿波": "green_wave", "自行车高速": "superhighway",
    "共享单车": "shared_bike", "头盔": "helmet", "盗窃": "theft",
    "伤害": "injury", "事故": "injury", "道路": "road", "路面": "road",
    "通勤": "commute", "出行占比": "modal_share", "出行比例": "modal_share",
    "出行总量": "modal_share", "交通": "transport", "基础设施": "infrastructure",
    "电动自行车": "ebike", "路口": "intersection", "信号系统": "signal_system",
    "投资": "investment", "执法": "enforcement",
    # --- sleep -----------------------------------------------------------
    "睡眠": "sleep", "快速眼动": "rem", "深睡眠": "deep_sleep",
    "深度睡眠": "deep_sleep", "周期": "cycle", "睡眠剥夺": "sleep_deprivation",
    "不睡": "sleep_deprivation", "睡眠不足": "sleep_deprivation",
    "睡不够": "sleep_deprivation", "低估": "underestimate", "受损": "impairment",
    "困倦": "sleepiness", "慢性": "chronic", "长期": "chronic",
    "酒精": "alcohol", "醉驾": "drunk_driving",
    "认知": "cognitive", "褪黑素": "melatonin", "松果体": "pineal_gland",
    "电子书": "ereader", "电子设备": "ereader", "屏幕": "screen",
    "青少年": "adolescent", "生物钟": "circadian", "昼夜节律": "circadian",
    "失眠": "insomnia", "睡眠限制": "sleep_restriction",
    "认知行为疗法": "cbt_i", "刺激控制": "stimulus_control",
    "睡眠效率": "sleep_efficiency", "午睡": "nap", "梦": "dream",
    # --- bees ------------------------------------------------------------
    "蜜蜂": "bee", "蜂群": "colony", "蜂王": "queen_bee", "工蜂": "worker_bee",
    "摇摆舞": "waggle_dance", "舞蹈": "dance", "花蜜": "nectar",
    "蜂蜜": "honey", "授粉": "pollination", "传粉": "pollination",
    "蜂群崩溃": "colony_collapse", "瓦螨": "varroa", "螨": "varroa",
    "农药": "pesticide", "扁桃": "almond", "杏仁": "almond",
    "农作物": "crop", "作物": "crop", "养蜂人": "beekeeper",
    "养蜂": "beekeeping", "采集蜂": "forager", "觅食": "foraging",
    "蜂箱": "hive", "蜂巢": "hive", "迁徙": "migration",
    "产卵": "egg", "卵": "egg", "春季": "spring", "花粉": "pollen",
    "温带": "temperate", "个体": "individual",
    # --- green space -----------------------------------------------------
    "热岛": "heat_island", "绿地": "green_space", "绿化": "green_space",
    "城市绿地": "green_space", "树冠": "canopy", "蒸腾": "transpiration",
    "降温": "cooling", "冷却": "cooling", "屋顶绿化": "green_roof",
    "屋顶花园": "green_roof", "透水铺装": "permeable_pavement",
    "透水": "permeable_pavement", "地表温度": "surface_temperature",
    "表面温度": "surface_temperature", "世界卫生组织": "who",
    "反射": "reflectance", "径流": "runoff", "公园": "park",
    "沥青": "asphalt", "混凝土": "concrete", "空调": "air_conditioner",
    "草地": "grass", "草坪": "grass", "遮阳": "shade", "遮荫": "shade",
    # --- generic relations ----------------------------------------------
    "方向": "direction", "距离": "distance", "角度": "angle", "时长": "duration",
    "温度": "temperature", "摄氏度": "celsius", "海拔": "altitude",
    "浓度": "concentration", "比例": "share", "占比": "share",
    "平均": "average", "年均": "annual_average", "每年": "annual",
    "研究": "study", "研究表明": "study", "实验": "experiment",
}

# --------------------------------------------------------------------------
# en surface form -> concept token(英文表面形式 → 概念词元)。
# 多词条目按短语先行匹配,再回落到单词级归一化;两侧最终落到同一个词元。
# --------------------------------------------------------------------------
EN: dict[str, str] = {
    # --- tea -------------------------------------------------------------
    "tea": "tea", "camellia sinensis": "tea", "tea plant": "tea",
    "green tea": "green_tea", "black tea": "black_tea", "oolong": "oolong",
    "pu-erh": "puerh", "puerh": "puerh",
    "polyphenol": "polyphenol", "polyphenols": "polyphenol",
    "catechin": "catechin", "catechins": "catechin",
    "caffeine": "caffeine", "theanine": "theanine",
    "fixation": "fixation", "rolling": "rolling", "drying": "drying",
    "fermentation": "fermentation", "fermented": "fermentation",
    "theaflavin": "theaflavin", "theaflavins": "theaflavin",
    "thearubigin": "thearubigin", "thearubigins": "thearubigin",
    "enzyme": "enzyme", "polyphenol oxidase": "ppo",
    "nitrogen": "nitrogen", "acidic soil": "acidic_soil",
    "shade": "shade", "shading": "shade", "humidity": "humidity",
    "storage": "storage", "yield": "output", "output": "output",
    # --- coffee ----------------------------------------------------------
    "coffee": "coffee", "roast": "roast", "roasts": "roast",
    "roasting": "roast", "roasted": "roast", "roaster": "roast",
    "green bean": "green_bean", "green beans": "green_bean",
    "first crack": "first_crack", "second crack": "second_crack",
    "light roast": "light_roast", "dark roast": "dark_roast",
    "maillard": "maillard", "maillard reaction": "maillard",
    "chlorogenic": "chlorogenic", "chlorogenic acid": "chlorogenic",
    "chlorogenic acids": "chlorogenic", "antioxidant": "antioxidant",
    "antioxidants": "antioxidant", "agtron": "agtron",
    "espresso": "espresso", "extraction": "extraction",
    "extract": "extraction", "pressure": "pressure", "bar": "pressure",
    "moisture": "moisture", "bean": "bean", "beans": "bean",
    "global production": "output", "world output": "output",
    # --- cycling ---------------------------------------------------------
    "bicycle": "bicycle", "bicycles": "bicycle", "bike": "bicycle",
    "bikes": "bicycle", "cycling": "cycling", "cyclist": "cyclist",
    "cyclists": "cyclist", "bicycle lane": "bike_lane",
    "bike lane": "bike_lane", "cycle track": "bike_lane",
    "cycle path": "bike_lane", "separation": "separation",
    "separated": "separation", "green wave": "green_wave",
    "superhighway": "superhighway", "bicycle superhighway": "superhighway",
    "bike sharing": "shared_bike", "shared bike": "shared_bike",
    "shared bicycle": "shared_bike", "helmet": "helmet",
    "helmets": "helmet", "theft": "theft", "injury": "injury",
    "injuries": "injury", "road": "road", "roads": "road",
    "commute": "commute", "commuting": "commute", "commuter": "commute",
    "commuters": "commute", "modal share": "modal_share",
    "share of trips": "modal_share", "transport": "transport",
    "infrastructure": "infrastructure", "intersection": "intersection",
    "intersections": "intersection", "traffic signal": "signal_system",
    "e-bike": "ebike", "electric bicycle": "ebike",
    # --- sleep -----------------------------------------------------------
    "sleep": "sleep", "rem": "rem", "rem sleep": "rem",
    "deep sleep": "deep_sleep", "cycle": "cycle", "cycles": "cycle",
    "sleep deprivation": "sleep_deprivation", "alcohol": "alcohol",
    "drunk driving": "drunk_driving", "cognitive": "cognitive",
    "underestimate": "underestimate", "underestimates": "underestimate",
    "underestimated": "underestimate", "impairment": "impairment",
    "impaired": "impairment", "sleepiness": "sleepiness", "sleepy": "sleepiness",
    "chronic": "chronic",
    "melatonin": "melatonin", "pineal": "pineal_gland",
    "pineal gland": "pineal_gland", "e-reader": "ereader",
    "ebook": "ereader", "screen": "screen", "teenager": "adolescent",
    "teenagers": "adolescent", "adolescent": "adolescent",
    "adolescents": "adolescent", "circadian": "circadian",
    "body clock": "circadian", "circadian rhythm": "circadian",
    "insomnia": "insomnia", "sleep restriction": "sleep_restriction",
    "cbt-i": "cbt_i", "cbt": "cbt_i", "stimulus control": "stimulus_control",
    "sleep efficiency": "sleep_efficiency", "nap": "nap", "naps": "nap",
    "dream": "dream", "dreaming": "dream",
    # --- bees ------------------------------------------------------------
    "bee": "bee", "bees": "bee", "honeybee": "bee", "honeybees": "bee",
    "apis mellifera": "bee", "colony": "colony", "colonies": "colony",
    "hive": "hive", "hives": "hive", "queen": "queen_bee",
    "queen bee": "queen_bee", "worker": "worker_bee",
    "worker bee": "worker_bee", "worker bees": "worker_bee",
    "waggle dance": "waggle_dance", "waggle": "waggle_dance",
    "dance": "dance", "nectar": "nectar", "honey": "honey",
    "pollination": "pollination", "pollinator": "pollination",
    "pollinators": "pollination", "pollinate": "pollination",
    "pollinated": "pollination", "colony collapse": "colony_collapse",
    "varroa": "varroa", "varroa destructor": "varroa", "mite": "varroa",
    "pesticide": "pesticide", "pesticides": "pesticide",
    "almond": "almond", "almonds": "almond", "crop": "crop",
    "crops": "crop", "beekeeper": "beekeeper", "beekeepers": "beekeeper",
    "beekeeping": "beekeeping", "forager": "forager",
    "foragers": "forager", "foraging": "foraging", "flight": "flight",
    "migration": "migration", "apiary": "apiary", "apiaries": "apiary",
    "egg": "egg", "eggs": "egg", "spring": "spring", "pollen": "pollen",
    "temperate": "temperate", "individual": "individual",
    "individuals": "individual", "production": "output",
    # --- green space -----------------------------------------------------
    "heat island": "heat_island", "urban heat island": "heat_island",
    "green space": "green_space", "green spaces": "green_space",
    "greenspace": "green_space", "canopy": "canopy", "tree canopy": "canopy",
    "transpiration": "transpiration", "evapotranspiration": "transpiration",
    "cooling": "cooling", "cool": "cooling", "cools": "cooling",
    "cooled": "cooling", "green roof": "green_roof",
    "green roofs": "green_roof", "roof garden": "green_roof",
    "permeable pavement": "permeable_pavement",
    "permeable paving": "permeable_pavement",
    "surface temperature": "surface_temperature",
    "world health organization": "who", "who": "who",
    "reflectance": "reflectance", "reflective": "reflectance",
    "runoff": "runoff", "park": "park", "parks": "park",
    "asphalt": "asphalt", "concrete": "concrete", "grass": "grass",
    "air conditioning": "air_conditioner", "air conditioner": "air_conditioner",
    "aroma": "aroma", "aromas": "aroma", "aromatic": "aroma",
    "brown": "brown", "investment": "investment", "invested": "investment",
    "invest": "investment",
    # --- generic relations ----------------------------------------------
    "direction": "direction", "distance": "distance", "angle": "angle",
    "duration": "duration", "temperature": "temperature",
    "celsius": "celsius", "degrees celsius": "celsius",
    "altitude": "altitude", "concentration": "concentration",
    "share": "share", "average": "average", "annual average": "annual_average",
    "study": "study", "studies": "study", "experiment": "experiment",
}

# --------------------------------------------------------------------------
# 实体名录:表面形式 → 规范实体 id。
# 用来阻止"讲实体 B 的段落去支持讲实体 A 的论断"(典型的张冠李戴)。
# --------------------------------------------------------------------------
ENTITIES: dict[str, str] = {
    "阿姆斯特丹": "amsterdam", "amsterdam": "amsterdam",
    "哥本哈根": "copenhagen", "copenhagen": "copenhagen",
    "新加坡": "singapore", "singapore": "singapore",
    "洛杉矶": "los_angeles", "los angeles": "los_angeles",
    "荷兰": "netherlands", "netherlands": "netherlands",
    "挪威": "norway", "norway": "norway",
    "丹麦": "denmark", "denmark": "denmark",
    "中国": "china", "china": "china",
    "美国": "usa", "united states": "usa",
    "北京": "beijing", "beijing": "beijing",
    "日本": "japan", "japan": "japan",
    "加州": "california", "加利福尼亚": "california",
    "california": "california", "越南": "vietnam", "vietnam": "vietnam",
    "巴西": "brazil", "brazil": "brazil",
    "澳大利亚": "australia", "australia": "australia",
    "浙江省": "zhejiang", "浙江": "zhejiang", "zhejiang": "zhejiang",
    "福建省": "fujian", "福建": "fujian", "fujian": "fujian",
    "云南省": "yunnan", "云南": "yunnan", "yunnan": "yunnan",
    "四川省": "sichuan", "四川": "sichuan", "sichuan": "sichuan",
}

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\.\-_]*")

# 货币名里包含国名("丹麦克朗"),不能当成国家实体,否则论断会被判成
# "提到了丹麦",从而把丹麦的证据错认成自家的证据。
_ENTITY_IGNORE = ("丹麦克朗", "人民币", "美元", "欧元", "日元", "英镑")


def _clean(tok: str) -> str:
    """去掉词元尾部的句点/连字符:``'grass.' -> 'grass'``。"""
    return tok.rstrip(".-_")

_EN_PHRASES = {k: v for k, v in EN.items() if " " in k}
_EN_SINGLE = {k: v for k, v in EN.items() if " " not in k}
_EN_PHRASE_RE = (
    re.compile(
        r"(?<![a-z0-9])(?:" +
        "|".join(re.escape(p) for p in sorted(_EN_PHRASES, key=len, reverse=True)) +
        r")(?![a-z0-9])")
    if _EN_PHRASES else None
)
_ZH_SORTED = sorted(ZH.items(), key=lambda kv: -len(kv[0]))
_ENT_PHRASES = {k: v for k, v in ENTITIES.items() if " " in k}
_ENT_SINGLE = {k: v for k, v in ENTITIES.items() if " " not in k and k.isascii()}
_ZH_ENTITIES = sorted(
    ((k, v) for k, v in ENTITIES.items() if not k.isascii()),
    key=lambda kv: -len(kv[0]))
_ENT_PHRASE_RE = (
    re.compile(
        r"(?<![a-z0-9])(?:" +
        "|".join(re.escape(p).strip() for p in _ENT_PHRASES) +
        r")(?![a-z0-9])")
    if _ENT_PHRASES else None
)


def concepts(text: str) -> set[str]:
    """返回文本的中英共享概念词元(不论原文是中文还是英文)。

    实体名也作为概念词元返回:``哥本哈根`` 与 ``Copenhagen`` 都映射到
    ``copenhagen``,这是跨语言召回里信息量最大的一类匹配特征。
    """
    low = text.lower()
    out: set[str] = set()
    if _EN_PHRASE_RE is not None:
        for m in _EN_PHRASE_RE.finditer(low):
            c = _EN_PHRASES.get(m.group(0))
            if c:
                out.add(c)
    for tok in _WORD.findall(low):
        c = _EN_SINGLE.get(_clean(tok))
        if c:
            out.add(c)
    for surface, concept in _ZH_SORTED:
        if surface in text:
            out.add(concept)
    out |= entities(text)
    return out


def entities(text: str) -> set[str]:
    """返回 *text* 中点名的实体 id 集合。"""
    for ignore in _ENTITY_IGNORE:
        text = text.replace(ignore, " " * len(ignore))
    low = text.lower()
    out: set[str] = set()
    if _ENT_PHRASE_RE is not None:
        for m in _ENT_PHRASE_RE.finditer(low):
            c = _ENT_PHRASES.get(m.group(0).strip())
            if c:
                out.add(c)
    for tok in _WORD.findall(low):
        c = _ENT_SINGLE.get(_clean(tok))
        if c:
            out.add(c)
    # 中文实体名没有空白分隔,只能按子串扫描。
    for surface, ent in _ZH_ENTITIES:
        if surface in text:
            out.add(ent)
    return out


__all__ = ["ZH", "EN", "ENTITIES", "concepts", "entities"]
