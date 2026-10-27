# -*- coding: utf-8 -*-
"""安全回归:仓库里不得出现真实密钥,密钥文件必须被忽略。

这是 AGENTS.md「密钥零入库」的可执行版本。任何一次提交前都应跑通。
"""
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", ".venv", "results", ".cache", "__pycache__",
             ".pytest_cache", "dist", "build", ".mypy_cache"}
TEXT_SUFFIXES = {".py", ".md", ".toml", ".txt", ".json", ".jsonl", ".yml",
                 ".yaml", ".cfg", ".ini", ".example", ""}

# 常见云服务密钥形态
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),          # OpenAI / DeepSeek
    re.compile(r"AKIA[0-9A-Z]{16}"),             # AWS access key
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),         # GitHub token
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),  # Slack token
]


def _repo_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name.startswith(".env.") and path.name != ".env.example":
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in (
                ".env", ".gitignore", ".env.example"):
            continue
        yield path


def test_no_real_secret_in_repo_files():
    offenders = []
    for path in _repo_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                offenders.append(f"{path.relative_to(ROOT)}: {match.group()[:8]}…")
    assert not offenders, "发现疑似真实密钥:\n" + "\n".join(offenders)


def test_env_is_placeholder_only():
    env = ROOT / ".env"
    assert env.exists(), ".env 应当存在(作为占位模板)"
    text = env.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("DEEPSEEK_API_KEY="):
            value = line.split("=", 1)[1].strip()
            assert value in ("", "sk-your-key-here"), "真实密钥不能写进 .env"
    assert not re.search(r"sk-[A-Za-z0-9]{20,}", text)


def test_env_example_documents_required_keys():
    example = ROOT / ".env.example"
    assert example.exists()
    text = example.read_text(encoding="utf-8")
    assert "DEEPSEEK_API_KEY" in text
    assert "FAITHSIFT_CAP" in text


def test_gitignore_covers_secret_files():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    lines = {line.strip() for line in text.splitlines()}
    assert ".env" in lines
    assert ".env.local" in lines
    assert ".venv/" in lines


def test_env_not_tracked_by_git():
    try:
        out = subprocess.run(
            ["git", "ls-files", "--error-unmatch", ".env"],
            cwd=ROOT, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        print("跳过:当前环境没有可用的 git")
        return
    assert out.returncode != 0, ".env 不应该被 git 跟踪"


def test_redaction_placeholder_not_flagged():
    """占位符不能被误报成密钥(否则测试本身就不可用)。"""
    placeholder = "sk-your-key-here"
    assert not any(p.search(placeholder) for p in SECRET_PATTERNS)
