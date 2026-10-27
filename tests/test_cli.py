# -*- coding: utf-8 -*-
"""CLI 行为与 CI 门禁退出码的离线测试。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from citecheck.cli import main

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "gold" / "sources"

ANSWER_OK = "茶树适宜生长的年均温度为 15 至 25 摄氏度。\n"
ANSWER_BAD = ("茶树适宜生长的年均温度为 15 至 25 摄氏度。\n"
              "绿茶加工中杀青温度通常控制在 180 摄氏度以上。\n"
              "中国茶叶在 2025 年的出口研发补贴预算约为 4500 万美元。\n")


def _write(tmp_path, text, name="answer.md"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_check_writes_three_formats(tmp_path):
    ans = _write(tmp_path, ANSWER_BAD)
    out = tmp_path / "report"
    rc = main(["check", str(ans), "--sources", str(SOURCES),
               "--out", str(out)])
    assert rc == 0
    for suffix in (".md", ".html", ".json"):
        assert out.with_suffix(suffix).exists()
    data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["summary"]["n_claims"] == 3
    assert data["meta"]["retriever"] == "lexical"


def test_check_json_output_is_machine_readable(tmp_path, capsys):
    ans = _write(tmp_path, ANSWER_OK)
    rc = main(["check", str(ans), "--sources", str(SOURCES),
               "--no-report", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["faithfulness"] == 1.0
    assert payload["verdicts"][0]["evidence_idxs"]


def test_fail_under_gate(tmp_path):
    ans = _write(tmp_path, ANSWER_BAD)
    rc = main(["check", str(ans), "--sources", str(SOURCES),
               "--no-report", "--fail-under", "0.99"])
    assert rc == 3, "忠实性低于阈值应返回 3"


def test_fail_under_passes_for_good_answer(tmp_path):
    ans = _write(tmp_path, ANSWER_OK)
    rc = main(["check", str(ans), "--sources", str(SOURCES),
               "--no-report", "--fail-under", "0.99"])
    assert rc == 0


def test_fail_on_contradicted_gate(tmp_path):
    ans = _write(tmp_path, ANSWER_BAD)
    rc = main(["check", str(ans), "--sources", str(SOURCES),
               "--no-report", "--fail-on-contradicted"])
    assert rc == 4, "出现与源冲突应返回 4"


def test_validate_min_accuracy_gate(capsys, tmp_path):
    out = tmp_path / "v.json"
    rc = main(["validate", "--dataset", "gold", "--limit", "20",
               "--min-accuracy", "1.01", "--out", str(out)])
    assert rc == 5
    assert out.exists()
    summary = json.loads(capsys.readouterr().out.split("\nfull -> ")[0])
    assert summary["n"] == 20


def test_validate_writes_result(tmp_path, capsys):
    out = tmp_path / "v.json"
    rc = main(["validate", "--dataset", "hard", "--limit", "5", "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["n"] == 5 and "by_category" in data


def test_version_exits_zero():
    try:
        main(["--version"])
    except SystemExit as e:
        assert e.code == 0
    else:
        raise AssertionError("--version 应当以 SystemExit(0) 结束")


def test_missing_gold_reports_usage_error(tmp_path, capsys):
    rc = main(["validate", "--gold", str(tmp_path / "nope.jsonl")])
    assert rc == 2
    assert "gold" in capsys.readouterr().err


def test_llm_mode_without_key_reports_cleanly(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("CITECHECK_ENV_FILE", raising=False)
    monkeypatch.setattr("citecheck.judge._load_api_key", lambda explicit=None: None)
    ans = _write(tmp_path, ANSWER_OK)
    rc = main(["check", str(ans), "--sources", str(SOURCES),
               "--mode", "llm", "--no-report"])
    assert rc == 6, "缺密钥应返回 6 并给出中文提示"
    assert "密钥" in capsys.readouterr().err
