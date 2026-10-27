# demo 产物说明

这个目录里的报告都是**真实跑出来的**,不是手写示意。注意区分版本与引擎。

| 文件 | 引擎 | 对应输入 | 说明 |
|---|---|---|---|
| `sample_answer.md` | — | — | 示例被检回答,故意混入三方问题:数字篡改、无据、正常支持 |
| `report.{md,html,json}` | `heuristic` | `sample_answer.md` | v0.2 生成,7 条论断:4 有据 / 1 无据 / 2 冲突,忠实性 57.1% |
| `multihop_answer.md` | — | — | 多跳 + 跨语言的示例输入 |
| `report_multihop.{md,html,json}` | `heuristic` | `multihop_answer.md` | v0.2 生成,3 条全对,其中 2 条被识别为多跳 |
| `report_llm.*` / `index.html` | `llm` | 旧版示例 | **v0.1 产物**,对应 184 条金标时期的提示词,保留作历史对照 |

重新生成(离线):

```bash
python -m faithsift check demo/sample_answer.md --sources gold/sources \
    --mode heuristic --out demo/report
python -m faithsift check demo/multihop_answer.md --sources gold/sources \
    --mode heuristic --out demo/report_multihop
```

重新生成 LLM 版(需要真实密钥,会产生费用):

```bash
python -m faithsift check demo/sample_answer.md --sources gold/sources \
    --mode llm --out demo/report_llm
```

`report_llm.*` 之所以没有在 v0.2 重跑,是因为仓库里不再存放真实密钥
(见 `AGENTS.md`「密钥零入库」)。请自行配置 `DEEPSEEK_API_KEY` 后重跑。
