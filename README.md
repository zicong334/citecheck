# FaithSift

**中文** | [English](README_en.md)

> 本项目原名 CiteCheck (renamed to FaithSift) —— 因名称冲突于 2026-10-27 更名,
> 缘由与证据见 [CHANGELOG](CHANGELOG.md)。旧 `CITECHECK_*` 环境变量仍作回退。

**RAG 回答的句子级引用忠实性检测器** —— 逐句判定 LLM 的回答是「有据 / 无据 / 与源冲突」,
并给出结论所依赖的证据定位。

给定一份回答(md/txt)和一组源文档,FaithSift 会把回答拆成句级论断,对每条:

1. **检索**:BM25 词法 + 数值不变性 + 中英概念词表(可选再接向量通道),找出最相关的证据片段;
2. **判定**三分类:**supported**(证据明确包含)/ **contradicted**(与同一事实的取值冲突)/
   **unsupported**(证据沉默,既不支持也不冲突);
3. **出报告**:逐条证据摘录、理由、整体忠实性得分,渲染 md / html / json。

`unsupported` 被单列出来是整套设计的核心:把「说错了」和「编了一个语料没覆盖的话题」分开,
工程上才能判断问题出在检索召回还是生成篡改。

## 实测成绩(诚实声明)

| 数据集 | 条数 | 启发式引擎 | 说明 |
|---|---|---|---|
| `gold/gold.jsonl`(合成) | 178 | **100.0%** | 词法友好,只作**回归下限** |
| `gold/hard_gold.jsonl`(人工) | 49 | **91.8%** | 改写 / 跨语言无数字 / 多跳 / 方向反转 / 实体错配 |

困难金标按类别的准确率:

| 能力 | 准确率 | 能力 | 准确率 |
|---|---|---|---|
| 实体错配(张冠李戴) | 100% | 数字篡改 | 100% |
| 多跳证据 | 100% | 单位错误 | 100% |
| 跨语言(无数字) | 87.5% | 中文同义改写 | 83.3% |
| 方向反转 | 80% | 相邻话题 vs 冲突 | 66.7% |

多跳证据召回 **7/7**。

**必须说清楚的三件事:**

1. 合成金标 100% 不代表真实水平。它由「原文抽取 / 数字篡改 / 模板套话」三类规则生成,
   天然对词法友好。它只保证重构不会把做对的事弄坏。
2. 困难金标是人工撰写的 49 条,每条都写了标注理由(`rationale` 字段),
   但它仍然是一人判断,不是多人标注的工业集。
3. **LLM 判官的分数尚未在 v0.2 提示词下复测**。仓库里的 `gold/validation_llm_v0.1.json`
   是 v0.1 提示词的结果(99.46%,¥0.29),对应的是 184 条旧金标。v0.2 改成了允许多跳数组,
   需要用真实密钥重跑 `python -m faithsift validate --dataset gold --mode llm`。

## 双引擎 × 三检索通道

| 维度 | 选项 | 依赖 | 用途 |
|---|---|---|---|
| 判官 | `heuristic` | 零 API、纯本地规则 | CI 门禁、离线快检、批量回归 |
| 判官 | `llm` | DeepSeek 结构化 JSON 判官 | 语义级判定:改写、蕴含陷阱、跨语言 |
| 检索 | `lexical` | 零依赖(BM25 + 数值 + 概念词表) | 默认通道 |
| 检索 | `hybrid` | 需嵌入后端,BM25+数值+向量融合 | 开放领域、重度改写 |
| 检索 | `dense` | 需嵌入后端,纯向量 + 数值 | 语义召回为主 |

向量通道永远不是必需品:没有 key 时 `--retriever hybrid` 会自动退回 `lexical`,
不会崩。离线替身 `--embedder hashing` 可以把整条链路的接线跑通(但它不跨语言)。

## 快速开始

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
python -m pytest tests/ -q                       # 70 项离线测试,零 API

python -m faithsift gold                         # 重建两套金标(确定性)
python -m faithsift validate --dataset gold      # 合成金标:回归下限
python -m faithsift validate --dataset hard      # 困难金标:能力缺口

# 核查一份回答
python -m faithsift check demo/sample_answer.md --sources gold/sources \
    --mode heuristic --out results/check
# -> results/check.{md,html,json}
```

`demo/report_multihop.md` 是多跳 + 跨语言召回的示例(3 条论断全对)。

## CLI

```text
faithsift check <answer> --sources <dir>
    --mode heuristic|llm|both         判官
    --retriever lexical|hybrid|dense  检索通道
    --embedder hashing|<模型名>        向量后端
    --top-k N --out results/check
    --fail-under 0.9                  忠实性低于阈值 -> 退出码 3
    --fail-on-contradicted            出现与源冲突 -> 退出码 4
    --json                            机器可读输出

faithsift gold                        重建 gold.jsonl 与 hard_gold.jsonl
faithsift validate --dataset gold|hard --mode heuristic|llm
    --min-accuracy 0.85               准确率低于阈值 -> 退出码 5
```

退出码:`0` 成功 · `2` 用法错误 · `3` 忠实性不达标 · `4` 存在冲突论断
· `5` 准确率不达标 · `6` 缺少密钥。

`--fail-under` / `--fail-on-contradicted` / `--min-accuracy` 就是挂在 CI 上的那道门禁。

## 判定标准(与 LLM 提示词一致)

- **supported**:证据明确包含论断的核心事实(实体与关系一致;非数值事实的同义改写、
  跨语言翻译算支持);需要多段证据才能成立时,把用到的片段编号全部列出。
- **contradicted**:证据就**同一事实**给出不同取值 —— 数字不符、方向相反(升高/降低、
  高于/低于)、实体张冠李戴、单位不同。
- **unsupported**:证据对论断主题保持沉默 —— 既不支持也不冲突。

三条硬规则:

1. **禁止数值逻辑蕴含**。证据说「200 度以上」、论断说「180 度以上」⇒ `contradicted`,
   哪怕逻辑上能推出来。改一个数字就是篡改。
2. **相邻话题不等于冲突**。语料讨论了邻近话题但没给论断所说的这个事实 ⇒ `unsupported`。
3. **声称支持必须给证据**。每条判定带 `evidence_idx`(多跳时为数组)。

## 几个值得一提的设计

**数字跨语言不变。** 数字不随翻译改变,所以「一爆 196 至 205 摄氏度」的中文论断能命中
英文段落。这一条把 LLM 引擎的 demo 从 5/8 提到 8/8。

**中文数词归一化。** 「两千」「四到六」「一成」「10 万」「1.9 million」「2,000」
都会先归一化成阿拉伯数字再比对,否则中文回答里的数词既进不了检索、也进不了数值校验。
`数百` 这类约数不会被误转成 100。

**概念词表补跨语言。** 只靠数字的话,一条不含数字的中文论断永远召回不到英文证据。
`faithsift/lexicon.py` 把中英表面形式映射到共享概念词元
(`茶树`/`tea` → `tea`,`一爆`/`first crack` → `first_crack`),
约 250 个表面形式、120 个概念,覆盖随附的六个领域。

**实体名录防张冠李戴。** 论断点名阿姆斯特丹,而「2005 年起投资 20 亿丹麦克朗」讲的是
哥本哈根 —— 数字全对、主题全对,但结论是错的。判官会把「只讲别的实体」的片段划为旁证,
数字全部落在旁证里就判 `contradicted`(实体错配)。

**多跳真的需要两段。** 判官不满足于「某段沾边」,只有在第二段证据带来足够新增覆盖、
并且合并后达到阈值时才判多跳支持;否则一篇主题相关但不含该事实的段落就会被当成证据。

**成本保险丝。** `results/cost_ledger.jsonl` 逐调用记账,硬顶 ¥12(`FAITHSIFT_CAP` 可调,旧名 `CITECHECK_CAP` 仍作回退),
到顶抛 `BudgetStop`,防止批量跑测试时超支。

## 结构

```text
faithsift/claims.py     中英句级论断拆分
faithsift/evidence.py   源加载 / 分块 / 分词
faithsift/lexicon.py    中英概念词表 + 实体名录
faithsift/numbers.py    中文数词与量级归一化
faithsift/retrieval.py  BM25 + 数值不变性 + 可选向量的混合检索
faithsift/embed.py      可选向量后端(OpenAI 兼容 / 哈希替身 / 磁盘缓存)
faithsift/judge.py      双引擎判官(LLM JSON 判官 + 启发式,含多跳)
faithsift/pipeline.py   端到端流水线 + 金标验证 + CI 门禁
faithsift/report.py     md / html / json 报告渲染
gold/                   两套金标 + 6 篇双语语料 + 验证产物
demo/                   示例回答与渲染好的报告
scripts/gap_report.py   按类别列出错分样本的开发脚本
tests/                  70 项离线测试
```

## 密钥与安全

真实密钥**不放进仓库**。加载顺序:

1. 环境变量 `DEEPSEEK_API_KEY`
2. `FAITHSIFT_ENV_FILE` 指向的文件(可以在仓库外;旧名 `CITECHECK_ENV_FILE` 仍作回退)
3. `.env.local`(已 gitignore,本地私有)
4. `.env`(只有占位符)

`tests/test_security.py` 会扫描仓库内文件是否出现真实密钥形态,并校验 `.env` 未被 git 跟踪。
CI 里单独跑这个文件。

> 如果真实密钥曾经出现在文件里,请到控制台**轮换**它,而不是只删文件 —— 已经分发出去的副本不受你控制。

## 已知局限(v0.2 仍未解决)

- **同语言重度改写**:仍主要依赖词法覆盖,「低估自己有多困」对「低估自己的受损程度」
  这类改写会漏判。这是向量通道该解决的问题。
- **相邻话题 vs 取值冲突**:「荷兰头盔立法始于 2010 年」语料讲的是澳大利亚 1990 年代,
  按定义是 `unsupported`,但词法引擎只看到「主题沾边 + 数字缺失」,会判成 `contradicted`。
- **无数字的语义方向**:依赖方向词表(提高/降低、促进/抑制),词表外的表达会漏。
- **概念词表只覆盖六个领域**:开放语料请用 `--retriever hybrid` 或 `--mode llm`。
- **金标规模小**:49 条困难样本足以暴露缺口,不足以给出置信区间。

## License

MIT。
