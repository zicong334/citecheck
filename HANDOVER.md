# HANDOVER — CiteCheck v0.1(2026-10-27 通宵构建)

## 现状(一句话)
RAG 引用忠实性检测器全链路完工:双引擎(LLM 判官 99.46% / 词法判官 97.83%,184 条确定性金标),
CLI + 三格式报告 + demo,8 项离线测试绿,LLM 全量验证仅花 ¥0.29,可直接挂 GitHub。

## 入口文件
- `README.md` — 对外门面(双引擎对照表+已知边界)
- `gold/build_gold.py` — 金标生成器(seed=20261027,确定性)
- `gold/validation_{heuristic,llm}.json` — 两套验证结果(含混淆矩阵+逐条明细)
- `demo/index.html` — LLM 引擎示例报告(8 条论断全判对)
- `demo/sample_answer.md` — 示例被检回答(故意混入三类问题)

## 复现命令
```bash
.venv/Scripts/python.exe -m pytest tests/ -q            # 8 绿(离线,零API)
.venv/Scripts/python.exe -m citecheck gold              # 重建金标(幂等)
.venv/Scripts/python.exe -m citecheck validate --mode heuristic
# LLM 模式读仓库根 .env / CITECHECK_ENV_FILE / 环境变量 DEEPSEEK_API_KEY(三选一)
.venv/Scripts/python.exe -m citecheck check demo/sample_answer.md --sources gold/sources --mode llm
```

## 关键设计决定
1. **三分类而非二分类**:unsupported(证据沉默)单列,区分"说错了"与"编了个对不上的话题"。
2. **数值信号跨语言检索**:retrieve() = 词法包含度 + 0.6×数字重合度——数字不随翻译改变,中文论断可检索到英文源(demo 里"一爆196-205"即此路径)。
3. **判官严禁数值蕴含**:证据"200度以上"≠论断"180度以上",一律 contradicted(提示词显式规定,修复前 LLM 会做逻辑推理放行)。
4. **成本保险丝**:CITECHECK_CAP 默认 ¥12,results/cost_ledger.jsonl 记账,到顶抛 BudgetStop。

## 已知局限 / v0.2 待办
- [ ] 金标是合成的(词法友好),需补真实 RAG 日志的人工标注集
- [ ] 启发式引擎单语限制:跨语言仅靠数字信号,无数字的跨语言论断会漏
- [ ] 检索是词法的:长文档场景应加向量检索通道(embedding)
- [ ] 多跳证据链(论断需要综合两段证据)未支持
- [ ] pip 发布 + GitHub Actions CI
- [ ] 与 ToolCallBench-CN 同属"LLM 可信输出"工具族,可互相引流

## 通宵执行记录
- 01:00 前后开工(与 B1 并行);语料 6 篇双语自写(零版权零网络)
- 首版启发式 40% 准确率(模板题误判)→ 重构为"主题覆盖×数值一致"双因子 → 97.8%
- demo 暴露检索跨语言缺陷(中文论断查不到英文源)→ 加数字加权检索 → LLM demo 8/8 全对
- LLM 判官金标 184 条 99.46%,花费 ¥0.29(远低于预估 ¥6,因论断短)
- 阻塞:无
