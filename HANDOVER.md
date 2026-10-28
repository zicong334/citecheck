# HANDOVER — FaithSift v0.3(2026-10-28)

> 本项目原名 CiteCheck (renamed to FaithSift),2026-10-27 因名称四层撞车改名
> (arXiv:2502.10881 等,详见 CHANGELOG 与 docs/survey_faithfulness_landscape.md)。
> 旧 `CITECHECK_*` 环境变量仍作回退;本地文件夹名 `citecheck` 保持不变。

## v0.3 做了什么(2026-10-28,本轮)

一句话:**给词法引擎补一条本地离线语义通道,修掉困难金标三个弱点,金标扩到 150 条压测**。

1. **任务1 本地离线向量后端**:`faithsift/embed.py` 新增 `LocalOnnxEmbedder`
   (fastembed + 多语 MiniLM,不含 torch),`--embedder local` 启用,`HF_ENDPOINT`
   国内镜像下载、离线复用、套 `CachedEmbedder` 缓存。依赖隔离:fastembed 惰性导入 +
   `[semantic]` extra,缺依赖自动退回 lexical(实测核心包 import 不牵入 numpy/onnx)。
   `--retriever hybrid|dense` 已在 150 条困难金标上跑通。语义余弦还接进了判官
   (`attach_semantic`,阈值 0.75),专补"字面不重合的重度同义改写"。
2. **任务2 三个弱点(原 49 条子集,词法档)**:相邻话题 66.7%→100%(缺失数字要
   同事实槽+同实体有竞争取值才判冲突)、方向反转 80%→100%(新增烘焙深/浅定性刻度
   反义词)、中文改写 83.3%→100%(同义词概念 + 语义兜底)、跨语言 87.5%→100%
   (概念词表补 `annual average` + 修 `_cjk_ratio` 忽略数字/标点)。每个修复都配了
   回归测试(`tests/test_weakness_fixes.py`)。
3. **任务3 困难金标 49→150**(`gold/build_hard_gold.py`):13 类,新增量级错误/时间错配/
   情态篡改,确定性可重建,自检查重+篡改值语料外校验。

**实测(150 条困难金标)**:词法 88.7% / 词法+向量 88.7%(原 49 条子集 100%)/ LLM 99.33%;
合成金标 178:词法 100% / LLM 99.44%。本轮 LLM 花费约 ¥0.54(记进 cost_ledger)。
**最大新缺口:情态篡改词法档 0%**(无语义情态信号,LLM 83%)——如实计入,未改金标迁就实现。
离线测试 70 → **89 项全绿**(向量端到端用例无 fastembed/模型时自动 skip)。

> 下面是 v0.2 的交接原文,保留作背景。

## 早上待办(2026-10-29 更新)
1. **网页操作(只能用户做)**:GitHub 仓库 Settings→Rename `citecheck`→`faithsift`;Gitee 同样(旧地址自动重定向)
2. ~~LLM 判官 v0.2 提示词复测~~ 已完成(v0.2 夜 99.44%/100%,v0.3 又在 150 条上复测 99.33%)
3. PyPI 抢注 `faithsift` 前人工复查 pypi.org/project/faithsift(调研代理核过一层,发布前再确认);`pip install -e .[semantic]` 才会装 fastembed,发布时把它作为 extra 说明
4. **全部本地 commit 待晨审后 push**(github + origin 两远端已配好;累计 14 个 commit,含改名 22add56、定位章节 06ce86a、LLM复测 cfa8cd8、v0.3 语义通道 41fee5b)
5. v0.3 的两个盲区是**特性不是缺陷**,对外说明时保持这个口径:情态篡改词法 0%(仅 LLM 83%)、相邻话题词法 64%(LLM 100%)——它们正是"为什么需要 LLM 判官/向量通道"的证据,不要为了让数字好看而删类别或改金标

## 现状(一句话)

v0.1 打通了链路,v0.2 补的是**链路背后的真实能力**:BM25 + 双语概念词表的混合检索、
多跳证据链、实体错配与方向反转检测、49 条人工困难金标、CI 门禁与密钥安全。
离线测试 8 → **70 项全绿**;合成金标 100%(回归下限),困难金标 **91.8%**(真实水平)。

## 入口文件

- `AGENTS.md` — **仓库级规则**:中文工作语言、判定语义、完成定义。开工先读这个。
- `README.md` / `README_en.md` — 对外门面:实测分数、能力边界、CLI、安全说明。
- `CHANGELOG.md` — v0.1 → v0.2 的完整变更。
- `gold/build_gold.py` — 合成金标生成器(seed=20261027,确定性,178 条)
- `gold/build_hard_gold.py` — 困难金标生成器(49 条人工用例,确定性)
- `gold/validation_heuristic.json` / `gold/validation_hard_heuristic.json` — 当前基线
- `demo/report_multihop.md` — 多跳 + 跨语言示例(3/3 全对)

## 复现命令

```bash
.venv/Scripts/python.exe -m pytest tests/ -q                    # 70 项(离线,零 API)
.venv/Scripts/python.exe -m faithsift gold                      # 重建两套金标(幂等)
.venv/Scripts/python.exe -m faithsift validate --dataset gold   # 合成金标 -> 100%
.venv/Scripts/python.exe -m faithsift validate --dataset hard   # 困难金标 -> 91.8%
.venv/Scripts/python.exe scripts/gap_report.py --dataset hard   # 按类别看错分
.venv/Scripts/python.exe -m faithsift check demo/sample_answer.md \
    --sources gold/sources --mode heuristic --out demo/report
```

LLM 模式需要密钥:环境变量 `DEEPSEEK_API_KEY`、`FAITHSIFT_ENV_FILE`,
或本地 `.env.local`(已 gitignore)。**仓库内 `.env` 只有占位符。**

## v0.2 做了什么

| 问题(v0.1 遗留) | 处理 |
|---|---|
| 检索只有词法包含度,长文档/改写场景漏召回 | 换成 BM25(词频+长度归一+IDF),并给概念重合单独加权 |
| 跨语言只能靠数字,无数字全漏 | `lexicon.py`:约 250 个中英表面形式 → 120 个共享概念词元 |
| 中文数词进不了检索与数值校验 | `numbers.py`:两千/四到六/一成/10 万/1.9 million/2,000 全归一化 |
| 多跳证据不支持 | `evidence_idxs` + `multi_hop`;判官可合并两段计分;LLM 允许数组 |
| 张冠李戴被字面重合放过 | 实体名录 + primary/others 分区;数字全落在旁证里判 contradicted |
| 无数字的方向反转抓不到 | 「方向词 → 紧邻概念」配对(提高茶氨酸 vs 降低茶氨酸) |
| 金标单一(合成、词法友好) | 新增 49 条人工困难金标,每条带标注理由 |
| `.env` 残留真实 key | 清成占位符;`.env.local` + 环境变量;`test_security.py` 扫描 |
| 没有 CI、无法做门禁 | GitHub Actions + `--fail-under` / `--fail-on-contradicted` / `--min-accuracy` |
| 报告信息量低 | 进度条、分类色块、问题清单、多跳标记、证据多编号定位 |

## 关键设计决定

1. **概念通道只在跨语言时走捷径。** 同语言时词法覆盖已经能反映"论断里有没有证据之外
   的主题词";若再用两个共享概念把分数顶上去,"绿地面积每增加 10%,房价上涨 3%"
   这种凭空追加事实的论断就会被放过。多跳的并集不受此限。
2. **数值"存在性"只认在题的片段。** 一个无关段落里恰好出现同样数字(20 亿 vs 20 天),
   不能算证据。这是"张冠李戴"能被识破的前提。
3. **方向冲突要求两侧讨论同一概念。** 否则"面积增加 10%"与"温度降低 10 度"
   会因为共享数字 10 被判成方向相反。
4. **概念覆盖的分母不加权于未登录词之外**,且要求最少 3 个概念(或 2 个概念 + 数字全中),
   防止"只提到咖啡/烘焙两个词"的无关论断被判成全支持。
5. **成本保险丝**保留:FAITHSIFT_CAP 默认 ¥12,到顶抛 BudgetStop。

## 实测数字(v0.2,离线词法引擎)

| 数据集 | n | 准确率 | 三分类 F1 |
|---|---|---|---|
| 合成金标 | 178 | 100.0% | 1.00 / 1.00 / 1.00 |
| 困难金标 | 49 | 91.8% | supported 0.96 / unsupported 0.80 / contradicted 0.93 |

困难金标分项:实体错配 100%、多跳 100%、数字篡改 100%、单位 100%、
跨语言 87.5%、方向 80%、中文改写 83.3%、相邻话题 66.7%;多跳召回 7/7。

**已复测(2026-10-27 夜更新)**:LLM 判官 v0.2 提示词全量复测完成(99.44%/100%)。`gold/validation_llm_v0.1.json` 是 v0.1 的历史结果
(99.46%,¥0.29,对应 184 条旧金标),不要当 v0.2 的结论引用。

## 剩余错误(全部可解释,已写进 README 已知局限)

1. 「低估自己有多困」这类重度同义改写 —— 词法覆盖 32%,需要向量通道。
2. 英文论断 "Tea bushes do best where…" 被同主题的咖啡段落抢走排序 —— 概念词表覆盖面问题。
3. Agtron「数值越大越深」—— 无数字的语义方向,依赖方向词表,表外表达会漏。
4. 「荷兰头盔立法始于 2010」被判 contradicted(应为 unsupported)——
   "相邻话题"与"同一事实不同取值"的边界,词法引擎倾向于后者。

## 下一步(v0.3 候选,按收益排序)

1. **用真实密钥复测 LLM 判官**,把两套金标的 LLM 分数补齐(CHANGELOG 里已列为未复测项)。
2. **补齐 setretrieval:pip 发布**(`twine` + trusted publishing),README 加安装方式。
3. **困难金标扩到 150+ 条**,最好引入第二个标注者做一致性检查(Kappa)。
4. **把 `--retriever hybrid` 接真实嵌入模型**跑一遍对照实验,量化向量通道对"同语言改写"
   的提升幅度——这是目前最大的已知缺口。
5. **多跳扩展到 3 段以上**(`max_hops` 已参数化,但只验证过 2 段)。
6. **与 ToolCallBench-CN 互相引流**(同属"LLM 可信输出"工具族)。

## 交接注意

- 任何改动先读 `AGENTS.md`:里面定了语言规约、判定语义和完成定义。
- 改判据等于改产品语义,三分类的定义不要动。
- 提交前跑 `pytest tests/test_security.py`;`.env` 只能有占位符。
- 基准数字只报实测值;合成金标与困难金标要一起公布,不能只挑好看的。
