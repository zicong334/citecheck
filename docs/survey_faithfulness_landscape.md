# 引用忠实性检测 Landscape 调研 + "CiteCheck" 名称冲突检查

- **调研日期**:2026-10-27
- **委托背景**:为自建 RAG 引用忠实性检测器(当前代号 CiteCheck,v0.2)撰写"与已发表工作的关系"章节做系统调研,并做名称冲突检查。
- **本工具特点基线**(用于重叠度评估):
  1. 回答拆句级论断 → 检索证据 → 三分类 supported / unsupported / contradicted(把"证据沉默"单列是卖点)
  2. 双引擎:零 API 词法判官(BM25 + 数值不变性 + 中英概念词表,合成金标 97.8%)+ DeepSeek 结构化 JSON 判官(99.46%)
  3. 确定性金标集(seed 固定;原文抽取 / 数字篡改 / 无关主题模板三类生成)
  4. 跨语言:数字信号检索 + 概念词表,中文论断命中英文源
  5. 判官提示词严禁数值逻辑蕴含("200度以上" ≠ "180度以上")
  6. 成本保险丝
  7. (v0.2)多跳证据链
  8. (v0.2)实体名录防张冠李戴

- **方法与诚实声明**:
  - bing MCP 服务器持续报 `Invalid request parameters`,不可用;WebSearch 工具按委托方提示未依赖。全部核实改用**直连官方 API**:arXiv Export API、GitHub REST API、PyPI JSON API、npm Registry、OpenAlex、HuggingFace Hub API,以及 WebFetch 抓官方文档页。
  - dblp API 有反爬(anubis bot check),venue 信息改用 OpenAlex DOI 元数据交叉核实。
  - Semantic Scholar API 触发限流(429),未使用。
  - 每条标注【已核实:URL】(本次实际抓取核实)或【未核实:仅训练记忆】。**查不到的写"未找到",不编造。**
  - 调研局限:未做引文图谱级的前向/后向检索(Semantic Scholar 限流、dblp 反爬),FacTool 后续工作、实体核验方向仅做了 arXiv 标题级关键词搜索,可能有遗漏。

---

## ① 逐项条目表

### 1. RAG 忠实性 / 归因评测框架

| 名称 | 作者/机构 | 年份 | Venue | 链接 | 核心机制一句话 | 与我们重叠度 |
|---|---|---|---|---|---|---|
| **RAGAS(faithfulness 指标)** | Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert(Cardiff University 等) | 2023 | EACL 2024 Demo(注意:**不是**常被引的 EMNLP 2023)【已核实:OpenAlex DOI 10.18653/v1/2024.eacl-demo.16】 | [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)【已核实】 / [GitHub](https://github.com/vibrantlabsai/ragas)(原 explodinggradients/ragas 已迁移,16k+ stars)【已核实:GitHub API 重定向】 | 无参考评估框架;faithfulness = 把回答拆成独立 statements,逐条判断能否从检索上下文推出,输出"可支持条数/总条数"的 0–1 连续分【已核实:https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/】 | **高**(拆句论断+对照证据完全同构);但它输出连续比例、不区分"沉默 vs 矛盾",判官依赖 LLM(文档提到可选 HHEM 分类器) |
| **ARES** | Jon Saad-Falcon 等(Stanford) | 2023 | NAACL 2024【已核实:OpenAlex DOI 10.18653/v1/2024.naacl-long.20】 | [arXiv:2311.09476](https://arxiv.org/abs/2311.09476)【已核实】 / [GitHub stanford-futuredata/ARES](https://github.com/stanford-futuredata/ARES)(740 stars)【已核实】 | 用少量人标 + PPI(预测幂推断)训练轻量判别器,给 RAG 的 context relevance / answer faithfulness / answer relevance 打自动分 | **中高**(同评 faithfulness,但走"训练判别器+统计校准"路线,需要标注种子;我们是零训练双引擎) |
| **TruLens** | TruEra(Snowflake 收购) | 2022– | 开源工具,无论文 | [GitHub truera/trulens](https://github.com/truera/trulens)(3.6k stars)【已核实】 / [文档](https://www.trulens.org/getting_started/core_concepts/feedback_functions/)【已核实】 | feedback functions 体系;groundedness 用本地 NLI 中等模型("groundedness NLI")对检索上下文验证 | **中高**(它也有"不依赖大模型 API"的 NLI 路线,与我们零 API 判官动机重叠;但 NLI 是神经网络模型,我们是纯词法 BM25+规则,且它无三分类语义标签、无跨语言设计) |
| **DeepEval** | Confident AI | 2023– | 开源工具,无论文 | [GitHub confident-ai/deepeval](https://github.com/confident-ai/deepeval)(19k stars)【已核实】 / [文档](https://deepeval.com/docs/metrics-faithfulness)【已核实】 | FaithfulnessMetric:LLM 抽 claims → 对照 retrieval context 判断,"truthful claims / total claims"比例分 | **高**(拆 claim 流程同构);注意其 hybrid 模式有 0.65/0.35 概率三档,但那是**置信度分档**而非"沉默/矛盾"语义分类——我们的语义三分类仍不同 |
| **Vectara HHEM** | Vectara | 2023– | 开源模型 + 排行榜,无正式论文(博客) | [GitHub vectara/hallucination-leaderboard](https://github.com/vectara/hallucination-leaderboard)(3.3k stars)【已核实】 / [HF 模型 vectara/hallucination_evaluation_model](https://huggingface.co/vectara/hallucination_evaluation_model)(102k 下载,370 likes)【已核实:HF API】 | HHEM(现 2.3 版)是专用幻觉评测分类模型,判断摘要与源文档的事实一致性,驱动 LLM 幻觉排行榜 | **低中**(仅摘要场景、二值一致性,非论断级、非检索引用场景;但常被引为"忠实性判官模型",related work 应提) |
| **LlamaIndex evaluation** | LlamaIndex | 2023– | 框架文档 | [文档](https://developers.llamaindex.ai/python/framework/module_guides/evaluating/)【已核实:WebFetch,原 docs.llamaindex.ai 301 至此】 | 提供 Correctness / Faithfulness / Context Relevancy / Answer Relevancy / Guideline Adherence 等 LLM-based 评估器 | **中**(Faithfulness 评估器同任务,纯 LLM 判官、无词法引擎、无三分类) |

### 2. 论断级事实核验

| 名称 | 作者/机构 | 年份 | Venue | 链接 | 核心机制一句话 | 与我们重叠度 |
|---|---|---|---|---|---|---|
| **FActScore** | Sewon Min 等(UW / Meta AI) | 2023 | EMNLP 2023 main【已核实:arXiv comment "Published as a main conference paper at EMNLP 2023"】 | [arXiv:2305.14251](https://arxiv.org/abs/2305.14251)【已核实】 / [GitHub shmsw25/FActScore](https://github.com/shmsw25/FActScore)(460 stars)【已核实】 | 把长文本拆成 atomic facts,逐条对可靠知识源(Wikipedia)算"被支持百分比" | **高**(原子论断分解范式的开创者之一);但知识源是 Wikipedia 而非用户自己的检索库,评估对象是人物传记等开放生成,非 RAG 引用忠实性 |
| **SAFE(DeepMind)** | Jerry Wei 等(Google DeepMind) | 2024 | NeurIPS 2024【已核实:OpenAlex venue "neural information processing systems" 2024】 | [arXiv:2403.18802](https://arxiv.org/abs/2403.18802)"Long-form factuality in large language models"【已核实】 / [GitHub google-deepmind/long-form-factuality](https://github.com/google-deepmind/long-form-factuality)(698 stars)【已核实】 | 长回答拆成单条事实,agent 化多步 Google 搜索自证,输出 supported / not supported(二值)+ F1@K 指标,附 LongFact 基准 | **高**(拆句+检索+判定同构);但 not supported 一类混装"沉默"与"矛盾",且强依赖搜索 API(与我们零 API 引擎相反) |
| **MiniCheck** | Liyan Tang 等(UT Austin / Bespoke Labs) | 2024 | EMNLP 2024【已核实:GitHub 仓库描述标注 [EMNLP 2024]】 | [arXiv:2404.10774](https://arxiv.org/abs/2404.10774)【已核实】 / [GitHub Liyan06/MiniCheck](https://github.com/Liyan06/MiniCheck)(230 stars;注意常见误写 liboran/MiniCheck 不存在)【已核实:GitHub 搜索】 | 400M 级小模型做"论断是否被 grounding 文档支持"的二值判定,成本比 GPT-4 判官低 400 倍、性能接近 | **高**(低成本专用判官与我们零 API/成本保险丝动机一致);但它是训练出来的神经模型、二值、英文为主 |
| **FacTool** | I-Ting(Amy)Chern 等(GAIR-NLP 等多机构) | 2023 | NeurIPS 2023【未核实:仅训练记忆;OpenAlex 只显示 arXiv 版,arXiv comment 为空】 | [arXiv:2307.13528](https://arxiv.org/abs/2307.13528)【已核实】 / [GitHub GAIR-NLP/factool](https://github.com/GAIR-NLP/factool)(936 stars)【已核实】 | 工具增强框架:抽论断 → 构造查询 → 调工具(搜索/代码执行/KB)取证 → 多数投票判真伪 | **中高**(抽论断+主动取证同构);但面向开放域多任务事实性,非 RAG 引用场景 |
| **FacTool 后续** | — | — | — | — | 本次仅做 arXiv 标题级检索,未做引文级追踪(Semantic Scholar 限流)。**未找到**系统性清单;相邻近期工作见下文 SemanticCite、claim normalization 等 | — |

相邻的近期工作(arXiv 搜索命中,标题与 ID 已核实,内容未逐篇细读):
- SemanticCite: Citation Verification with AI-Powered Full-Text Analysis and Evidence-Based Reasoning,arXiv:2511.16198(2025-11)【已核实:arXiv API】
- Understanding before verifying: Claim normalization for automated citation verification,arXiv:2608.30145(2026-08)【已核实:arXiv API】
- Rethinking Inline Citation Verification in Scholarly Communication,arXiv:2610.05355(2026-10)【已核实:arXiv API】

### 3. 引用 / 归因基准

| 名称 | 作者/机构 | 年份 | Venue | 链接 | 核心机制一句话 | 与我们重叠度 |
|---|---|---|---|---|---|---|
| **ALCE** | Tianyu Gao 等(Princeton) | 2023 | EMNLP 2023 main【已核实:OpenAlex DOI 10.18653/v1/2023.emnlp-main.398;GitHub 描述同标】 | [arXiv:2305.14627](https://arxiv.org/abs/2305.14627)"Enabling LLMs to Generate Text with Citations"【已核实】 / [GitHub princeton-nlp/ALCE](https://github.com/princeton-nlp/ALCE)(528 stars)【已核实】 | 首个带引用生成的自动评测基准;fluency / correctness / citation quality 三维,citation recall & precision 用 NLI 蕴含模型对(句子,引文)打分 | **高**(句子级"引文是否支持该句"就是同一任务);其 NLI 判官是现成蕴含模型(三分类模型取 entailment 概率),没有把"沉默"作为面向用户的独立标签;我们金标里"数字篡改"类对应它的负样本构造思路 |
| **ExpertQA** | Chaitanya Malaviya 等(UPenn) | 2023 | NAACL 2024【已核实:OpenAlex DOI 10.18653/v1/2024.naacl-long.167】 | [arXiv:2309.07852](https://arxiv.org/abs/2309.07852)【已核实】 | 领域专家撰写问题+评估带引用回答的归因质量(attributed answer) | **中**(归因评估基准,但面向专家问答内容质量,非自动化忠实性判官) |
| **CRAG(Meta)** | Xiao Yang, Kai Sun 等(Meta) | 2024 | NeurIPS 2024 Datasets & Benchmarks【已核实:arXiv comment】 | [arXiv:2406.04744](https://arxiv.org/abs/2406.04744)"CRAG -- Comprehensive RAG Benchmark"【已核实】 | 4409 问答对的综合 RAG 基准,覆盖 5 领域 8 问题类型,含动态变化事实与多跳,配 mock API 检索 | **中**(是 RAG 系统基准而非引用忠实性判官;其多跳/动态事实题目与我们 v0.2 多跳证据链方向相关) |
| **RGB** | Jiawei Chen 等(浙江大学等) | 2023 | AAAI 2024【已核实:OpenAlex DOI 10.1609/aaai.v38i16.29728】 | [arXiv:2309.01431](https://arxiv.org/abs/2309.01431)"Benchmarking LLMs in Retrieval-Augmented Generation"【已核实】 / [GitHub chen700564/RGB](https://github.com/chen700564/RGB)(378 stars)【已核实】 | 从噪声鲁棒性/负样本拒绝/信息整合/反事实鲁棒性四能力面测 RAG;**含中文数据集**(中英双语) | **中**(负样本拒绝≈对"证据沉默/无关"的处理能力,中英双语与我们跨语言相关;但它是模型能力基准,不是判官工具) |
| **Attributed QA** | Bohnet 等(Google) | 2022 | arXiv 预印本(OpenAlex 未见正式 venue)【已核实:arXiv+OpenAlex】 | [arXiv:2212.08037](https://arxiv.org/abs/2212.08037)"Attributed Question Answering: Evaluation and Modeling for Attributed LLMs"【已核实】 | 提出 Attributed QA 任务与评估框架,以人工归因标注为金标、找相关自动指标 | **中高**(归因评估的奠基工作;其人工金标+自动指标相关性验证的方法论,正是我们合成金标想低成本替代的东西) |
| **AIS(理论基础)** | Hannah Rashkin 等(Google) | 2021/2023 | Computational Linguistics 2023【已核实:OpenAlex DOI 10.1162/coli_a_00486】 | [arXiv:2112.12870](https://arxiv.org/abs/2112.12870)"Measuring Attribution in Natural Language Generation Models"【已核实;注意网上常传的 2112.11446 是错误 ID(那是 Gopher 论文)】 | 定义"attributable to identified sources"的可解释归因框架与人类评估协议 | **中**(概念基础,写 related work 的锚点;非工具) |
| **CAQA** | (记忆中存在此基准) | — | — | — | **未找到**:arXiv ti:"CAQA" 与 all:"citation quality assessment" 检索均未命中对应论文,记忆中 ID 2401.11943 核实为无关论文。放弃收录,不编造 | — |

### 4. 三分类(supported / unsupported / contradicted,"证据沉默"单列)是否有人做过?

**结论:三分类范式本身在事实核验与 NLI 领域有明确先例,不算首创;但在"RAG 引用忠实性判官工具"这一具体生态里,主流产品仍是二值或 0–1 连续分,把"证据沉默"作为面向用户的独立语义标签的现成工具较少——这是可以主张的差异点,但措辞必须收敛为"产品化组合差异"而非"范式首创"。**

先例(均已核实):
- **FEVER**(Thorne 等,NAACL 2018,[arXiv:1803.05355](https://arxiv.org/abs/1803.05355)):claim 三分类 **Supported / Refuted / NotEnoughInfo**——"证据不足"单列的直接鼻祖【已核实:arXiv API,摘要原文明确三分类】
- **NLI 传统**(SNLI/MNLI/**XNLI**,Conneau 等 2018,[arXiv:1809.05053](https://arxiv.org/abs/1809.05053)):entailment / **neutral** / contradiction 三分,neutral 即"证据沉默"【已核实】
- **ALCE**:用三分类 NLI 模型算 citation recall/precision(取 entailment),但最终指标不向用户暴露"沉默 vs 矛盾"的区分【已核实:论文+repo】
- **L-MARS**(Yuan & Wang,AI4Law Workshop @ ICML 2026,[arXiv:2509.00761](https://arxiv.org/abs/2509.00761)):法律 QA 逐 claim 审计,用**六类标签体系** + strict-ALCE 跨供应商判官——说明"比三分类更细"的方向已有人走【已核实:arXiv API,摘要】
- **Evaluating and Guarding Citation Faithfulness in Agentic Scientific Synthesis**([arXiv:2607.20527](https://arxiv.org/abs/2607.20527),2026-07):实证不同严格度的 verifier 对同一输出的 unsupported 率给出 3%~18% 的差异,且各 verifier 在"supported"上一致、在"flag 什么"上不一致——**这直接支持我们"判官本身需要确定性金标来审计"的动机**,强烈建议在 related work 引用【已核实:arXiv API,摘要】
- 反例(二值/连续):RAGAS(比例分)、DeepEval(比例分)、SAFE(supported/not)、MiniCheck(二值)、HHEM(一致性二值)【均已核实,见上表】

### 5. 中文 / 跨语言忠实性检测

| 名称 | 年份 | 链接 | 一句话 | 与我们关系 |
|---|---|---|---|---|
| **CiteCheck(北大,Xu et al.)** | 2025 | [arXiv:2502.10881](https://arxiv.org/abs/2502.10881)【已核实】 / [GitHub xzy-xzy/CiteCheck](https://github.com/xzy-xzy/CiteCheck)【已核实:README 抓取】 | **首个大规模中文引用忠实性数据集**;两阶段人工标注+LLM 增强负样本(ch=信息改写/del=信息删除),**二值标签**(0/1),含微调小模型实验;作者含 Zhijiang Guo、Houfeng Wang(北大) | **最直接的学术撞车**:同任务(citation faithfulness detection)+ 同语言(中文)+ 同名。差异:它是数据集+微调判别模型,我们是免训练双引擎工具+三分类+跨语言(中文论断→英文源);其负样本构造(ch/del)与我们金标三类生成(篡改/无关模板)思路部分重合。venue:OpenAlex 仅显示 arXiv(2025),正式发表处**未查到** |
| **HalluQA** | 2023 | [arXiv:2310.03368](https://arxiv.org/abs/2310.03368)"Evaluating Hallucinations in Chinese Large Language Models"【已核实】 | 中文 LLM 幻觉问答基准(误导性/长尾问题) | 中文幻觉评测先例,但非引用忠实性、非检索场景 |
| **XNLI** | 2018 | [arXiv:1809.05053](https://arxiv.org/abs/1809.05053)【已核实】 | 跨语言 NLI 基准(15 语言) | "中文前提蕴含英文假设"这类跨语言判定的学术基础 |
| **CrossHallu** | 2026 | [arXiv:2607.04029](https://arxiv.org/abs/2607.04029)【已核实:arXiv 搜索命中,内容未细读】 | 幻觉信号是否跨语言/跨域泛化 | 跨语言幻觉检测新工作,related work 可引 |
| **PsiloQA** | 2025 | [arXiv:2510.04849](https://arxiv.org/abs/2510.04849)【已核实:同上】 | 多语言 span 级幻觉检测 | 同上 |
| **MultiHaluDet** | 2026 | [arXiv:2605.24919](https://arxiv.org/abs/2605.24919)【已核实:同上】 | 多语言幻觉检测(hidden-state probing) | 同上 |
| **ANHALTEN** | 2024 | [arXiv:2407.13702](https://arxiv.org/abs/2407.13702)【已核实:同上】 | 德语 token 级无参考幻觉检测(跨语言迁移) | 同上 |
| RGB(中英双语) | 2023 | 见第 3 节 | RAG 能力基准含中文 | 中文 RAG 评测先例 |

**未找到**:与"中文论断 → 数字信号+概念词表检索 → 命中英文源"完全相同的现成管道。这个**跨语言证据检索**设计在检索到的文献里没有直接对应物,是可主张的差异点(但应引用 XNLI/CrossHallu 说明跨语言判定有学术脉络,且 CiteCheck@北大 已占住"中文引用忠实性"叙事高地)。

### 6. 名称冲突检查(重点)——结论:**严重冲突,必须改名**

#### 6.1 arXiv 论文(3 篇同名)

| 论文 | ID | 日期 | 是什么 |
|---|---|---|---|
| **CiteCheck: Towards Accurate Citation Faithfulness Detection** | [arXiv:2502.10881](https://arxiv.org/abs/2502.10881) | 2025-02-15 | 北大等;首个大规模**中文引用忠实性数据集**(见第 5 节)。**同任务、同语言、同名——最致命冲突**【已核实:arXiv API + GitHub README】 |
| **CiteCheck: Retrieval-Grounded Detection of LLM Citation Hallucinations in Scientific Text** | [arXiv:2605.27700](https://arxiv.org/abs/2605.27700) | 2026-05-26(v2) | Khajavi, Sadeghi, Adhikari, Tessier;检索外部学术源验证引文是否真实存在、元数据是否忠实的混合框架(引文幻觉检测)【已核实:arXiv API,摘要】 |
| **citecheck: An MCP Server for Automated Bibliographic Verification and Repair in Scholarly Manuscripts** | [arXiv:2603.17339](https://arxiv.org/abs/2603.17339) | 2026-03-18 | Junhyeok Lee;TypeScript MCP server,书目元数据校验与修复【已核实:arXiv API,摘要】 |

#### 6.2 GitHub(API 搜索 `citecheck`,共 **69** 个仓库命中,前 30 已核)【已核实:GitHub Search API,2026-10-27】

与我们定位**直接相撞**的(不止一个!):
- [junaidshahid-dev/citecheck](https://github.com/junaidshahid-dev/citecheck) — "Audit RAG answers for citation faithfulness — flags claims that aren't actually supported by the cited…"(几乎是我们的一句话简介)
- [amaanmithani/citecheck](https://github.com/amaanmithani/citecheck) — "Checks whether citations in RAG answers support their sentences — with the checker itself validated…"
- [kushwanthpallenna-alt/citecheck](https://github.com/kushwanthpallenna-alt/citecheck) — "AI-powered citation verification platform that checks whether research claims are supported by their…"(2026-10-26 仍在更新)
- [lucasgris/citecheck](https://github.com/lucasgris/citecheck) — MCP server,对 LaTeX/Markdown 稿中引文核对被引段落
- [Narwinh/citecheck](https://github.com/Narwinh/citecheck) — 多 agent 研究助手,逐句引用+逐 claim 对源验证
- [ankitkumartiwari-web/citecheck](https://github.com/ankitkumartiwari-web/citecheck) — RAG 论文助手,claim 级引用验证
- [ult666666/citecheck](https://github.com/ult666666/citecheck) — 逐句引用+checker 验证
- [Jivraj-18/citecheck](https://github.com/Jivj-18/citecheck)(正确链接:[Jivraj-18](https://github.com/Jivraj-18/citecheck)) — LLM 引用检查的 accuracy vs cost 研究
- [xzy-xzy/CiteCheck](https://github.com/xzy-xzy/CiteCheck) — 上述北大论文的官方数据/代码仓

其他同名(领域不同但占名):
- [color4-alt/CiteCheck](https://github.com/color4-alt/CiteCheck)(68 stars,最活跃)— Agent Skill:查学术论文引用的格式/可查询性/主题相关性/语义准确性,2026-10-26 更新
- [tobiasosDev/citecheck](https://github.com/tobiasosDev/citecheck) — .bib/.ris/CSL-JSON 参考文献存在性与撤稿检查(npm 包同源)
- [nathanjmcdougall/citecheck](https://github.com/nathanjmcdougall/citecheck) — Python 引用链运行时保护(即 PyPI 上的 citecheck)
- [marchoag/citecheck](https://github.com/marchoag/citecheck) — 判例法引用检查;[PierreBeaucoral/citecheck](https://github.com/PierreBeaucoral/citecheck) — 学术 PDF 引用完整性(撤稿/伪造/掠夺性期刊);[Rushant-123/citecheck](https://github.com/Rushant-123/citecheck) — 按调用售卖的 agent 引用完整性检查;等

#### 6.3 包管理器

- **PyPI `citecheck`:已被占用**(0.3.0,"Run-time protection of citation chains",nathanjmcdougall,有 readthedocs 文档站)【已核实:https://pypi.org/pypi/citecheck/json 返回 200】;`cite-check` 为 404(可用)【已核实】
- **npm `citecheck`:已被占用**(Tobias Lüscher,zotero/bibtex/crossref/retraction 检查工具,latest 1.4.0)【已核实:https://registry.npmjs.org/citecheck】

#### 6.4 结论

"CiteCheck / citecheck / Cite-Check"在**论文(3 篇,其中 1 篇是同任务同语言的中文工作)、GitHub(69 仓,其中约 10 个与 RAG 引用忠实性直接同定位)、PyPI、npm** 四个层面全部被占,且撞名对象里有活跃更新的项目。**继续使用此名不可行**,发表时会造成引用混淆,发包时会直接冲突。

#### 6.5 改名建议(每个候选均按同样流程查过占用,2026-10-27)

| 候选名 | PyPI | npm | GitHub | arXiv | 评价 |
|---|---|---|---|---|---|
| **FaithSift** ✅推荐 | 404 可用 | 404 可用 | 0 仓 | 0 命中 | 四层全干净;语义贴合("忠实性筛分"),sift 呼应三分类筛选 |
| **SilentJudge** ✅推荐 | 404 可用 | 404 可用 | 0 仓 | 0 命中 | 四层全干净;直接点题"证据沉默"这一核心卖点(evidence-silent 也归 judge 管) |
| **CiteSift** | 404 可用 | 404 可用 | 1 仓([alexanderfurnas/CiteSift](https://github.com/alexanderfurnas/CiteSift),OpenAlex 注释书目生成,领域相邻但功能不同) | 0 命中 | 基本干净;保留 "Cite" 前缀的延续性,但有 1 个相邻领域小仓,存在混淆残余风险 |
| (落选)ClaimSentry | 404 | 404 | 5 仓(保险理赔反欺诈方向,领域不同) | 0 | 名字被保险科技占了心智,不建议 |
| (落选)AttriScope | 404 | 404 | 3 仓(其中 1 个是 LM token 归因研究工具,领域相邻) | 0 | 相邻撞名,不建议 |
| (落选)ClaimTrace | **200 已占** | 404 | 36 仓 | 0 | PyPI 被占,出局 |

以上占用状态均【已核实:对应 API 直查】。注意 GitHub 搜索结果时效性强,改名落地前建议再复查一次。

---

## ② 重叠度总结

**红海(不要再当卖点写的)**:
1. "回答拆句级论断 → 检索/对照证据 → 逐条判定"是 2023 年以来的标准范式:FActScore、SAFE、FacTool、RAGAS、DeepEval、ALCE 全部如此。这一条只能写成"遵循既有范式",不能作为贡献。
2. LLM 判官 + 结构化输出:DeepEval/TruLens/RAGAS 均已产品化。
3. 低成本判官动机:MiniCheck(小模型)、TruLens(本地 NLI)、HHEM(专用小模型)都占了"不用 GPT-4 当判官"的叙事。

**可以主张的差异点(按强度排序,均已对照上文核实)**:
1. **纯词法零 API 判官引擎**(BM25 + 数值不变性 + 中英概念词表,金标 97.8%):现有低成本方案全是"训练/下载一个神经模型"(MiniCheck、HHEM、TruLens-NLI),**没有找到**"完全不跑神经模型、纯词法+规则"的现成对标物。这是最干净的差异点,且与 arXiv:2607.20527 揭示的"神经判官之间互相不一致"问题形成呼应。
2. **确定性合成金标(seed 固定、三类生成)用于审计判官本身**:2607.20527 证明"审计判官的可靠性"是真问题;FEVER/CiteCheck(北大)的负样本构造是人工或 LLM 增强,**未找到**"seed 可复现 + 数字篡改 + 无关模板"同款设计。
3. **跨语言证据命中(中文论断 → 英文源,数字信号+概念词表)**:未找到直接对应管道;但"中文引用忠实性"叙事已被北大 CiteCheck(2502.10881)占住,写作时必须正面引用并划清边界(他们=中文数据集+微调判别器;我们=免训练工具+跨语言检索+三分类)。
4. **面向用户的三分类语义标签("证据沉默"单列)**:范式上 FEVER/NLI 早有三分,产品上主流是比例分/二值;可主张"在 RAG 引用判官工具中把 NEI 语义显式暴露给终端用户+金标专门覆盖沉默类",但**严禁**写成"首次提出三分类"。
5. **数值蕴含禁令**("200度以上"≠"180度以上")与**数值不变性**检索信号:在检索范围内未找到把它做成显式机制的工作(MiniCheck 论文摘要层面未见此细节);算小而实的工程贡献。
6. **(v0.2)多跳证据链**:事实核验侧有 FEVEROUS(NeurIPS 2021 D&B,多跳+表格)【已核实:arXiv comment】,RAG 基准侧有 CRAG(含多跳题)【已核实】;引用忠实性**判官工具**做多跳证据链的未找到,可主张,但要引 FEVEROUS 说明学术脉络。
7. **(v0.2)实体名录防张冠李戴**:仅找到间接相邻(L-MARS 六类 taxonomy 细节未核实是否含实体错配类;SemanticCite 做全文级验证)。**未找到**直接对标;可主张但检索置信度中等(未做引文级追踪)。
8. **成本保险丝**:纯工程特性,无学术对标必要。

**related work 必引清单**(全部已核实):RAGAS、ALCE、FActScore、SAFE、MiniCheck、FacTool、ARES、FEVER、AIS(Rashkin)、Attributed QA(Bohnet)、RGB、CRAG、HalluQA、CiteCheck@北大(2502.10881)、verifier 可靠性(2607.20527)、L-MARS(2509.00761)。

---

## ③ 名称冲突结论与改名建议(摘要)

- **CiteCheck 不可用**:3 篇 arXiv 同名论文(含同任务同语言的北大中文引用忠实性数据集)+ 69 个 GitHub 同名仓库(约 10 个与 RAG 引用忠实性直接同定位,部分活跃更新)+ PyPI、npm 双占名。
- **推荐改名**:首选 **FaithSift** 或 **SilentJudge**(四层全干净,2026-10-27 核实);次选 **CiteSift**(仅 1 个相邻领域 GitHub 小仓)。落地前复查一次占用。

---

## ④ 引用清单(带核实状态)

### 已核实(本次调研实际抓取)
| 条目 | URL | 核实方式 |
|---|---|---|
| RAGAS 论文 | https://arxiv.org/abs/2309.15217 | arXiv API(标题/作者/摘要) |
| RAGAS venue = EACL 2024 demo | DOI 10.18653/v1/2024.eacl-demo.16 | OpenAlex |
| RAGAS GitHub(迁移后) | https://github.com/vibrantlabsai/ragas | GitHub API(重定向解析) |
| RAGAS faithfulness 机制 | https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/ | WebFetch(HTTP 200) |
| ARES 论文 | https://arxiv.org/abs/2311.09476 | arXiv API |
| ARES venue = NAACL 2024 | DOI 10.18653/v1/2024.naacl-long.20 | OpenAlex |
| ARES GitHub | https://github.com/stanford-futuredata/ARES | GitHub API |
| TruLens GitHub | https://github.com/truera/trulens | GitHub API |
| TruLens groundedness NLI | https://www.trulens.org/getting_started/core_concepts/feedback_functions/ | WebFetch |
| DeepEval GitHub | https://github.com/confident-ai/deepeval | GitHub API |
| DeepEval faithfulness 机制 | https://deepeval.com/docs/metrics-faithfulness | WebFetch |
| Vectara 排行榜 | https://github.com/vectara/hallucination-leaderboard | GitHub API + WebFetch |
| HHEM 模型 | https://huggingface.co/vectara/hallucination_evaluation_model | HF API |
| LlamaIndex evaluation | https://developers.llamaindex.ai/python/framework/module_guides/evaluating/ | WebFetch(原 docs.llamaindex.ai 301 至此) |
| FActScore 论文(EMNLP 2023) | https://arxiv.org/abs/2305.14251 | arXiv API(comment 含 venue) |
| FActScore GitHub | https://github.com/shmsw25/FActScore | GitHub API |
| SAFE 论文 | https://arxiv.org/abs/2403.18802 | arXiv API |
| SAFE venue = NeurIPS 2024 | OpenAlex venue 字段 | OpenAlex |
| SAFE GitHub | https://github.com/google-deepmind/long-form-factuality | GitHub API |
| MiniCheck 论文 | https://arxiv.org/abs/2404.10774 | arXiv API |
| MiniCheck GitHub(EMNLP 2024) | https://github.com/Liyan06/MiniCheck | GitHub 搜索 API |
| FacTool 论文 | https://arxiv.org/abs/2307.13528 | arXiv API |
| FacTool GitHub | https://github.com/GAIR-NLP/factool | GitHub 搜索 API |
| ALCE 论文 | https://arxiv.org/abs/2305.14627 | arXiv API(摘要全文) |
| ALCE venue = EMNLP 2023 | DOI 10.18653/v1/2023.emnlp-main.398 | OpenAlex |
| ALCE GitHub | https://github.com/princeton-nlp/ALCE | GitHub API |
| ExpertQA 论文 | https://arxiv.org/abs/2309.07852 | arXiv API |
| ExpertQA venue = NAACL 2024 | DOI 10.18653/v1/2024.naacl-long.167 | OpenAlex |
| CRAG 论文(NeurIPS 2024 D&B) | https://arxiv.org/abs/2406.04744 | arXiv API(comment) |
| RGB 论文 | https://arxiv.org/abs/2309.01431 | arXiv API |
| RGB venue = AAAI 2024 | DOI 10.1609/aaai.v38i16.29728 | OpenAlex |
| RGB GitHub | https://github.com/chen700564/RGB | GitHub API |
| Attributed QA 论文 | https://arxiv.org/abs/2212.08037 | arXiv API(摘要) |
| AIS 论文 | https://arxiv.org/abs/2112.12870 | arXiv API |
| AIS venue = Computational Linguistics 2023 | DOI 10.1162/coli_a_00486 | OpenAlex |
| FEVER 论文(三分类) | https://arxiv.org/abs/1803.05355 | arXiv API(摘要含 Supported/Refuted/NotEnoughInfo) |
| FEVEROUS(NeurIPS 2021 D&B) | https://arxiv.org/abs/2106.05707 | arXiv API(comment) |
| XNLI | https://arxiv.org/abs/1809.05053 | arXiv API |
| CiteCheck@北大 论文 | https://arxiv.org/abs/2502.10881 | arXiv API(摘要全文) |
| CiteCheck@北大 GitHub/数据格式 | https://github.com/xzy-xzy/CiteCheck | raw README 抓取 |
| CiteCheck(科学文本引文幻觉) | https://arxiv.org/abs/2605.27700 | arXiv API(摘要) |
| citecheck MCP server 论文 | https://arxiv.org/abs/2603.17339 | arXiv API(摘要) |
| verifier 可靠性论文 | https://arxiv.org/abs/2607.20527 | arXiv API(摘要) |
| L-MARS(AI4Law @ ICML 2026) | https://arxiv.org/abs/2509.00761 | arXiv API(comment+摘要) |
| HalluQA | https://arxiv.org/abs/2310.03368 | arXiv API |
| 幻觉综述(Huang et al.) | https://arxiv.org/abs/2311.05232 | arXiv API |
| CrossHallu / PsiloQA / MultiHaluDet / ANHALTEN / SemanticCite / claim normalization / inline citation verification | arXiv:2607.04029 / 2510.04849 / 2605.24919 / 2407.13702 / 2511.16198 / 2608.30145 / 2610.05355 | arXiv API 搜索命中(仅标题+ID,内容未细读) |
| GitHub "citecheck" 69 仓命中清单 | https://api.github.com/search/repositories?q=citecheck | GitHub Search API |
| PyPI citecheck 占用 | https://pypi.org/pypi/citecheck/json | HTTP 200 + 元数据 |
| npm citecheck 占用 | https://registry.npmjs.org/citecheck | JSON 元数据 |
| 改名候选占用状态(FaithSift/SilentJudge/CiteSift/ClaimSentry/AttriScope/ClaimTrace) | PyPI/npm/GitHub/arXiv 四查 | 各官方 API,2026-10-27 |

### 未核实(仅训练记忆,写作引用前需人工复核)
- FacTool venue = NeurIPS 2023(OpenAlex/arXiv 均只显示预印本信息)
- RAGAS 作者机构细节(Cardiff University / 企业归属变动)
- L-MARS 六类 taxonomy 的具体类别构成(摘要未展开)
- DeepEval hybrid 模式阈值语义的更多细节(以文档页为准,已初步核实)

### 未找到(诚实记录)
- CAQA 引用质量基准:记忆中的线索未能核实,放弃收录
- "FacTool 后续"系统清单:未做引文级追踪
- 与"词法判官+数值不变性+概念词表"完全对标的现成工具:未找到
- 与"中文论断→英文源跨语言证据检索"完全对标的现成管道:未找到
