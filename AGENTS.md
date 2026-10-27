# AGENTS.md — 仓库级协作约定(v0.2)

> 本文件是项目的**底层规则**,优先级高于任何单次对话里的临时措辞。
> 任何在本仓库工作的 AI 代理或协作者,开工前必须先读本文件。
>
> 项目原名 CiteCheck (renamed to FaithSift),2026-10-27 改名;历史文档里的
> CiteCheck 均指本项目。包/命令/导入一律用 `faithsift`。

## 1. 语言规约(强制)

**本仓库的工作语言是中文。** 具体落地到以下每一层:

| 场景 | 要求 |
|---|---|
| 思考过程(推理、计划、自查) | **一律中文**。不写英文思路再翻译。 |
| 对话回复、任务汇报、总结 | **一律中文**。术语首次出现可括注英文原词。 |
| 代码注释、模块/函数文档字符串(docstring) | **一律中文**。 |
| 报错信息、CLI 帮助文本、日志 | **中文优先**;保留必要的英文标识符和错误码。 |
| 提交信息(commit message)、PR 描述、CHANGELOG | **一律中文**。 |
| 测试用例名、断言失败提示 | 用例名可用英文标识符(便于 pytest 收集),断言里的说明文字用中文。 |
| 文档(README、HANDOVER、docs/\*) | 中文优先;`README_en.md` 是唯一面向海外读者的英文文件,与中文 README 同步维护。 |

**不受此约束的部分**(保持英文是行业惯例,强行改中文反而有害):

- 变量名、函数名、类名、模块名、包名、CLI 子命令与参数名;
- 第三方库 API 名称、HTTP 头、环境变量名(如 `DEEPSEEK_API_KEY`);
- JSON 字段名(`verdict`/`label`/`faithfulness` 等)与三分类标签值
  (`supported`/`unsupported`/`contradicted`)——这些是跨语言接口契约,改了会破坏兼容性;
- 代码块内的实际代码。

## 2. 判定语义(不得随意改动)

三分类是本项目的核心资产,改判据等于改产品:

- `supported`:证据明确包含论断核心事实;非数值事实的同义改写、跨语言翻译算支持。
- `contradicted`:证据就**同一事实**给出不同取值——数字不符、方向相反、实体错配、单位不同。
- `unsupported`:证据对论断**保持沉默**——既不支持也不冲突。

补充硬规则:

1. **禁止数值逻辑蕴含**。证据"200 度以上"、论断"180 度以上" ⇒ `contradicted`,
   不得因为"逻辑上能推出"而放行。
2. **相邻话题不等于冲突**。证据讨论了邻近话题但没给论断所说的这个事实 ⇒
   `unsupported`,不是 `contradicted`。
3. **声称支持必须给证据**。每条判定都要带 `evidence_idx`(可为数组,多跳时列出全部)。

## 3. 工程约定

- **离线优先**:`heuristic` 引擎与全部单元测试必须零网络、零 API key 可跑。
  任何联网能力(LLM 判官、向量模型)都必须是可选通道,缺 key 时给出明确中文报错并降级。
- **零重量级依赖**:核心链路只用标准库。新增依赖必须写进 `pyproject.toml` 的
  optional extras,并说明为什么不能不用。
- **诚实报告**:基准数字只报实测值,并注明数据集与复现命令。禁止挑选好看的子集。
  合成金标是"回归下限",困难金标是"能力缺口",两者都要公布。
- **密钥零入库**:仓库内任何文件都不得出现真实 API key。`.env` 只放占位符,
  真实密钥走环境变量或本地 `.env.local`(已 gitignore)。提交前跑 `pytest tests/test_security.py`。
- **可复现**:金标生成必须确定性(固定随机种子),报告产物可重建。

## 4. 完成定义(Definition of Done)

一个改动算"完成",需要同时满足:

1. `python -m pytest tests/ -q` 全绿(离线);
2. `python -m faithsift validate --dataset gold` 准确率不低于既有基线(当前 97.8%);
3. 新增能力在 `gold/hard_gold.jsonl` 上有可解释的分数变化,并在 HANDOVER 里注明;
4. 相关文档(README / HANDOVER / CHANGELOG)同步更新,且用中文;
5. 没有新增网络依赖到离线路径。

## 5. 目录职责

```
faithsift/claims.py     中英句级论断拆分
faithsift/evidence.py   源加载 / 分块 / 分词(含双语概念词表)
faithsift/lexicon.py    中英概念词表 + 实体名录(零依赖的跨语言召回基础)
faithsift/numbers.py    中文数词 → 阿拉伯数字归一化
faithsift/retrieval.py  BM25 / 数值不变性 / 向量通道的混合检索
faithsift/embed.py      可选向量后端(OpenAI 兼容 / 哈希占位 / 磁盘缓存)
faithsift/judge.py      双引擎判官(LLM JSON 判官 + 词法启发式,含多跳)
faithsift/pipeline.py   端到端流水线 + 金标验证 + CI 门禁
faithsift/report.py     md / html / json 报告渲染
gold/                   合成金标(回归下限) + 困难金标(能力缺口) + 双语语料
scripts/                开发辅助脚本(不参与打包)
tests/                  离线测试
```
