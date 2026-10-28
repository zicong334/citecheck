# -*- coding: utf-8 -*-
"""本地离线向量后端(LocalOnnxEmbedder)的测试。

分两层:
  * 不依赖 fastembed 的测试(别名解析、缺依赖时自动退回 lexical)——CI 必过;
  * 依赖已缓存模型的端到端跨语言测试——fastembed 未装或模型未缓存时自动 skip,
    绝不联网下载(用 local_files_only=True)。
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faithsift import embed as embed_mod
from faithsift.embed import (DEFAULT_LOCAL_MODEL, LocalOnnxEmbedder,
                             SemanticBackendUnavailable)
from faithsift.evidence import discover_sources, load_sources
from faithsift.retrieval import build_retriever

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "gold" / "sources"


def _chunks():
    return load_sources(discover_sources(SOURCES))


# --------------------------------------------------------------------------
# 不需要 fastembed 的部分
# --------------------------------------------------------------------------
def test_local_model_alias_resolution():
    assert LocalOnnxEmbedder._resolve("local") == DEFAULT_LOCAL_MODEL
    assert LocalOnnxEmbedder._resolve("multilingual-minilm") == DEFAULT_LOCAL_MODEL
    assert LocalOnnxEmbedder._resolve(
        "local:intfloat/multilingual-e5-small") == "intfloat/multilingual-e5-small"
    # 非别名的完整模型名原样透传
    assert LocalOnnxEmbedder._resolve(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ) == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def test_semantic_backend_unavailable_is_runtime_error():
    """降级异常必须是 RuntimeError 子类,才能被现有 CLI 的 except 捕获。"""
    assert issubclass(SemanticBackendUnavailable, RuntimeError)


def test_build_retriever_falls_back_to_lexical_when_backend_unavailable(
        monkeypatch, capsys):
    """缺 fastembed / 模型加载失败 -> 中文提示 + 自动退回 lexical,不崩。"""
    def boom(self, *a, **k):
        raise SemanticBackendUnavailable("模拟:未安装 fastembed")

    monkeypatch.setattr(embed_mod.LocalOnnxEmbedder, "__init__", boom)
    ret = build_retriever(_chunks(), backend="hybrid", embedder_model="local")
    assert ret.backend == "lexical", "缺后端时 hybrid 应降级为 lexical"
    assert ret.dense is None
    err = capsys.readouterr().err
    assert "faithsift" in err and "词法" in err
    # 降级后仍能正常检索
    assert ret.search("茶树 15 至 25 摄氏度")


def test_core_import_does_not_require_fastembed():
    """依赖隔离硬要求:不装 fastembed 也能 import 核心包并用词法引擎。

    这里不真的卸载 fastembed,而是断言 embed 模块顶层没有 import fastembed
    (惰性导入在 __init__ 内),且 lexical 检索链路完全不触碰它。
    """
    src = (ROOT / "faithsift" / "embed.py").read_text(encoding="utf-8")
    # 顶层不得出现 import fastembed(只允许在函数/方法体内惰性导入)
    top_level = [ln for ln in src.splitlines()
                 if ln.startswith("import fastembed")
                 or ln.startswith("from fastembed")]
    assert not top_level, "fastembed 必须是惰性导入,不能出现在模块顶层"
    # 纯词法检索不需要任何向量后端
    ret = build_retriever(_chunks(), backend="lexical", embedder_model=None)
    assert ret.search("咖啡一爆 196 至 205 摄氏度")


# --------------------------------------------------------------------------
# 需要已缓存模型的部分(fastembed 未装 / 模型未缓存 -> skip,绝不联网)
# --------------------------------------------------------------------------
def _local_or_skip():
    try:
        emb = LocalOnnxEmbedder("local", local_files_only=True)
        emb.encode(["ping"])  # 触发模型加载
        return emb
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"本地向量后端不可用(未装 fastembed 或模型未缓存): {e}")


def test_local_embedder_crosslingual_similarity():
    emb = _local_or_skip()
    vecs = emb.encode(["茶树原产于中国西南地区",
                       "Tea plants originated in southwest China",
                       "阿姆斯特丹的自行车盗窃率"])
    assert len(vecs) == 3
    assert len(vecs[0]) == 384
    assert all(isinstance(x, float) for x in vecs[0]), "必须是纯 python float,不泄漏 numpy"

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b)) / (
            math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

    # 中英同义句的相似度,应显著高于中文句与无关句
    assert cos(vecs[0], vecs[1]) > cos(vecs[0], vecs[2]) + 0.1


def test_local_embedder_reuses_disk_cache(tmp_path):
    """套上 CachedEmbedder 后,重复文本不再重新过模型(命中磁盘缓存)。"""
    emb = _local_or_skip()

    class Counting:
        model = emb.model

        def __init__(self):
            self.calls = 0

        def encode(self, texts):
            self.calls += 1
            return emb.encode(texts)

    inner = Counting()
    cache = embed_mod.CachedEmbedder(inner, path=tmp_path / "emb.json")
    a = cache.encode(["茶树 15 至 25 摄氏度"])
    b = cache.encode(["茶树 15 至 25 摄氏度"])
    assert inner.calls == 1, "第二次应命中缓存"
    assert a == b
    assert (tmp_path / "emb.json").exists()


def test_hybrid_channel_end_to_end_on_hard_gold():
    """打通 --retriever hybrid + --embedder local 在困难金标上的验证链路。

    并验证向量通道的**增量价值**:一条字面几乎不重合的重度同义改写,纯词法
    覆盖不足会漏判,接入语义通道后应被判为 supported。
    """
    _local_or_skip()  # 模型不可用则 skip
    from faithsift.pipeline import validate_against_gold
    hard = ROOT / "gold" / "hard_gold.jsonl"
    res = validate_against_gold(hard, SOURCES, mode="heuristic",
                                retriever="hybrid", embedder="local",
                                progress=False)
    assert res["retriever"] == "hybrid"
    assert res["accuracy"] >= 0.85, res["by_category"]
    # 语义通道补上纯词法的短板:重度同义改写不再漏判
    assert res["by_category"]["paraphrase_zh"]["accuracy"] >= 0.90, res["by_category"]


def test_semantic_channel_fixes_heavy_paraphrase():
    """纯词法漏判的重度改写,接入向量通道后判为 supported(语义兜底)。"""
    emb = _local_or_skip()
    from faithsift.embed import CachedEmbedder, DenseIndex
    from faithsift.judge import HeuristicJudge
    from faithsift.retrieval import Retriever
    chunks = load_sources(discover_sources(SOURCES))
    ret = Retriever(chunks, backend="hybrid",
                    dense=DenseIndex(chunks, CachedEmbedder(emb)))
    judge = HeuristicJudge()
    judge.attach_idf(ret.bm25.idf)
    judge.attach_semantic(ret.dense)
    claim = "成年人一夜大约会经历四到六个睡眠周期,每个周期九十分钟上下。"
    v = judge.judge(claim, [c for c, _ in ret.search(claim, k=3)])
    assert v.verdict == "supported", v.reason
    assert "语义" in v.reason or v.score >= 0.55, v.reason
