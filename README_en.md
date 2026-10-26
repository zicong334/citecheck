# CiteCheck

[中文](README.md) | **English**

Sentence-level **citation-faithfulness checker for RAG answers**. Feed it an answer (md/txt) plus the source documents the RAG pipeline retrieved from; it splits the answer into claims and labels each one:

- **supported** — the sources clearly contain this fact (paraphrase & translation tolerant)
- **contradicted** — the sources state something conflicting (numbers must match *exactly*; no logical-entailment free passes)
- **unsupported** — the sources are silent: no evidence either way

`unsupported` as a first-class label is the point: most hallucination checkers are binary, conflating "got it wrong" with "made up a topic the sources never covered". Treating them separately is what makes RAG hallucination governance actionable.

Output: per-claim verdict table with evidence quotes + reasons, an overall faithfulness score (supported ratio), rendered as **md / html / json**.

## Dual engines

| Engine | Dependency | Accuracy on deterministic gold (184 claims, 3-class balanced) | Use case |
|---|---|---|---|
| `heuristic` | zero-API lexical + numeric-consistency rules | **97.8%** (F1: 1.00 / 0.97 / 0.96) | offline quick checks, CI gates |
| `llm` | DeepSeek structured-JSON judge | **99.5%** (F1: 0.99 / 0.99 / 1.00) | semantic judgments: paraphrase, entailment traps, cross-lingual |

Full LLM validation over all 184 gold claims cost **¥0.29 (~$0.04)**.

**Honest caveats** (reproducible via the demo): the gold set is synthetic and lexical-friendly — it is a regression floor, not a substitute for human eval. The heuristic engine is monolingual-lexical: cross-lingual claims survive only on numeric signals (numbers don't translate); use `--mode llm` for semantic cross-lingual checking.

## Retrieval trick: numbers are language-agnostic

Chunk scoring = lexical containment + **0.6 × distinctive-number overlap**. A Chinese claim about "一爆 196–205°C" retrieves the English paragraph containing 196/205 with zero shared words. This single change moved the LLM-engine demo from 5/8 to 8/8 correct verdicts.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/pip install openai python-dotenv pytest
python -m pytest tests/ -q                    # 8 offline tests

python -m citecheck gold                      # rebuild gold set (deterministic, seed=20261027)
python -m citecheck validate --mode heuristic # lexical engine vs gold
python -m citecheck validate --mode llm       # LLM engine vs gold (needs key, ~¥0.3)

python -m citecheck check demo/sample_answer.md --sources gold/sources \
    --mode heuristic --out results/check      # -> results/check.{md,html,json}
```

LLM mode needs a DeepSeek key: `DEEPSEEK_API_KEY` env var, or `CITECHECK_ENV_FILE` pointing at a `.env`. Cost fuse: append-only ledger `results/cost_ledger.jsonl`, hard cap ¥12 (`CITECHECK_CAP`), raises `BudgetStop` on breach.

## Judge contract (prompt-encoded)

The LLM judge is explicitly forbidden from logical entailment on numbers: source says "above 200°C", claim says "above 180°C" → **contradicted** (tampered value), even though it's logically implied. Non-numeric facts may be paraphrased or translated. Verdicts must cite the evidence chunk index and quote it.

## Layout

```
citecheck/claims.py     zh/en sentence-level claim splitter
citecheck/evidence.py   source loading, chunking, numeric-boosted retrieval (zero deps)
citecheck/judge.py      LLM judge (robust JSON parse, backoff, ledger) + heuristic judge
citecheck/pipeline.py   check_answer / validate_against_gold
citecheck/report.py     md/html/json rendering
gold/                   deterministic gold builder + 6 bilingual source docs + validation artifacts
demo/                   sample answer + rendered reports (index.html = LLM-engine demo)
```

## Status & license

MIT. v0.1 built overnight 2026-10-26/27. Roadmap: real RAG-log replay with human-labelled gold, NLI small-model engine (zero-API semantic), vector retrieval channel for long corpora, multi-hop evidence chains, pip release + CI.

Sister project: [ToolCallBench-CN](../toolcallbench-cn) — same "trustworthy LLM output" tool family, different failure surface (tool-calling vs citation faithfulness).
