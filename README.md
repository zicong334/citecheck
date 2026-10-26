# CiteCheck

**RAG 引用忠实性检测器** — 逐句判定 LLM 回答是"有据 / 无据 / 与源冲突",并给出证据定位。

给定一份回答(md/txt)和一组源文档,CiteCheck 把回答拆成句级论断(claims),对每条论断:
1. 词法检索最相关的证据片段(中文按字符二元组、英文按词,零分词依赖);
2. 判定三分类:**supported**(证据明确包含)/ **contradicted**(与证据冲突)/ **unsupported**(证据中无依据);
3. 输出证据摘录、理由与整体忠实性得分(supported 占比),渲染 md / html / json 三种报告。

## 双引擎
| 引擎 | 依赖 | 合成金标准确率* | 用途 |
|---|---|---|---|
| `heuristic` | 零 API,纯词法+数值一致性规则 | **97.8%**(三类 F1: 1.00/0.97/0.96) | 离线快检、CI 门禁 |
| `llm` | DeepSeek(结构化 JSON 判官) | 见 `results/gold_validation_llm.json` | 语义级判定(改写、蕴含) |

\* 在 184 条三类平衡合成金标上测得(`gold/build_gold.py` 确定性生成:原文抽取=supported、数字篡改=contradicted、无关主题模板=unsupported)。**诚实声明**:合成金标偏词法友好,是回归下限而非人类评估替代品;真实 RAG 场景的改写/蕴含判定请结合 `--mode both` 对照使用。

## 快速开始
```bash
python -m venv .venv && .venv/Scripts/pip install openai python-dotenv pytest
python -m pytest tests/ -q                      # 8 项离线自检

python -m citecheck gold                        # 重建金标集(确定性,seed=20261027)
python -m citecheck validate --mode heuristic   # 词法引擎 vs 金标
python -m citecheck validate --mode llm         # LLM 引擎 vs 金标(需 key,约¥2)

# 检查一份回答
python -m citecheck check demo/sample_answer.md --sources gold/sources \
    --mode heuristic --out results/check
# -> results/check.{md,html,json}
```
LLM 模式需要 DeepSeek key:环境变量 `DEEPSEEK_API_KEY`,或 `CITECHECK_ENV_FILE` 指向 .env。
成本保险丝:`results/cost_ledger.jsonl` 逐调用记账,硬顶 ¥12(`CITECHECK_CAP` 可调),到顶抛 `BudgetStop`。

## 判定标准(与 LLM 判官提示词一致)
- **supported**:证据明确包含论断核心事实;同义改写算支持;
- **contradicted**:同一事实、不同取值(数字不符/方向相反/实体张冠李戴);
- **unsupported**:证据对论断主题保持沉默——既不支持也不冲突。

第三类是 RAG 幻觉治理最容易被忽略的:模型说得头头是道,但检索源里根本没有——
把"无据"单列出来,才能区分"说错了"和"编了一个对不上的话题"。

## 结构
```
citecheck/claims.py     中英句级论断拆分
citecheck/evidence.py   源加载/分块/双语词法检索(零依赖)
citecheck/judge.py      LLM 判官(JSON 鲁棒解析+退避+记账) + 启发式判官
citecheck/pipeline.py   check_answer / validate_against_gold
citecheck/report.py     md/html/json 报告渲染
gold/                   确定性金标生成器 + 6 篇双语源语料(茶/骑行/咖啡/睡眠/蜜蜂/绿地)
demo/                   静态示例报告
```

## License & Status
MIT. v0.1(2026-10-27 通宵构建)。后续路线:真实 RAG 日志回放、NLI 小模型引擎、多跳证据链核验。
