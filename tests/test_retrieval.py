# -*- coding: utf-8 -*-
"""检索层离线测试:BM25、数值不变性、跨语言概念召回、可选向量通道。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faithsift.embed import CachedEmbedder, DenseIndex, HashingEmbedder
from faithsift.evidence import (Chunk, discover_sources, distinctive_numbers,
                                load_sources, tokenize)
from faithsift.lexicon import concepts, entities
from faithsift.retrieval import BM25Index, Retriever, build_retriever, numeric_hit

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "gold" / "sources"


def _chunks():
    return load_sources(discover_sources(SOURCES))


def test_tokenize_keeps_legacy_tokens():
    toks = tokenize("茶多酚占干重18%的Maillard反应")
    assert "茶多" in toks and "maillard" in toks and "18" in toks


def test_sentence_final_punctuation_stripped():
    """句末的 grass. 必须和 grass 视作同一个词,否则英文概念永远匹配不上。"""
    assert "grass" in tokenize("plain grass.")
    assert "grass" in concepts("plain grass.")


def test_concepts_are_symmetric():
    """同一概念的中英表述要落到同一个词元上。"""
    for zh, en in [("茶树", "tea"), ("一爆", "first crack"), ("哥本哈根", "Copenhagen"),
                   ("褪黑素", "melatonin"), ("树冠", "tree canopy"),
                   ("屋顶绿化", "green roofs")]:
        assert concepts(zh) & concepts(en), f"{zh} 与 {en} 应有共享概念"


def test_entities_and_currency_guard():
    assert entities("阿姆斯特丹的自行车") == {"amsterdam"}
    assert entities("Amsterdam and Copenhagen") == {"amsterdam", "copenhagen"}
    # 『丹麦克朗』是货币名,不能把论断变成"提到了丹麦"
    assert "denmark" not in entities("阿姆斯特丹自 2005 年起投资 20 亿丹麦克朗")


def test_bm25_ranks_exact_chunk_first():
    chunks = _chunks()
    idx = BM25Index(chunks)
    scores = idx.raw_scores(list(tokenize("咖啡一爆 first crack")))
    best = chunks[max(range(len(scores)), key=lambda i: scores[i])]
    assert best.doc == "coffee_en.md"
    assert all(s >= 0 for s in scores)


def test_crosslingual_recall_without_any_digit():
    """中文论断、英文证据、零阿拉伯数字:只能靠概念通道召回。"""
    chunks = _chunks()
    ret = Retriever(chunks)
    claim = "蜜蜂用摇摆舞把食物所在的方向和距离告诉同伴。"
    hits = ret.search(claim, k=3)
    assert hits, "应当至少召回一个片段"
    assert hits[0][0].doc == "bees_en.md", hits[0][0].doc
    assert distinctive_numbers(claim) == set(), "该用例不应含任何数字"


def test_numeric_invariance_retrieval():
    """数字跨语言不变:中文论断里的 196/205 应命中英文段落。"""
    chunks = _chunks()
    ret = Retriever(chunks)
    hits = ret.search("咖啡烘焙的一爆发生在豆温 196 至 205 摄氏度", k=2)
    assert hits[0][0].doc == "coffee_en.md"
    assert "196" in hits[0][0].text


def test_numeric_hit_ratio():
    chunk = Chunk(doc="x", idx=0, text="first crack reaches 196 to 205 degrees")
    assert numeric_hit("196 至 205 摄氏度", chunk) == 1.0
    assert numeric_hit("196 至 208 摄氏度", chunk) == 0.5
    assert numeric_hit("没有数字", chunk) == 0.0


def test_retriever_scores_are_positive_and_sorted():
    chunks = _chunks()
    ret = Retriever(chunks)
    hits = ret.search("茶多酚占干重 18% 至 36%", k=3)
    assert 0 < len(hits) <= 3
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)


def test_unknown_backend_rejected():
    try:
        Retriever(_chunks(), backend="magic")
    except ValueError as e:
        assert "magic" in str(e)
    else:
        raise AssertionError("未知后端应当报错")


def test_dense_channel_plumbing():
    """向量通道是可插拔的:用零依赖的哈希嵌入即可验证整条链路。"""
    chunks = _chunks()
    emb = HashingEmbedder(dim=128)
    dense = DenseIndex(chunks, emb)
    assert dense.stats()["n_chunks"] == len(chunks)
    assert dense.stats()["finite"] is True
    sims = dense.cosine("coffee roasting first crack")
    assert len(sims) == len(chunks)

    ret = build_retriever(chunks, backend="hybrid", embedder_model="hashing")
    hits = ret.search("coffee roasting first crack", k=3)
    assert hits and hits[0][0].doc == "coffee_en.md"


def test_hybrid_without_dense_falls_back_to_lexical():
    chunks = _chunks()
    ret = Retriever(chunks, backend="hybrid", dense=None)
    assert ret.backend == "lexical"
    assert ret.search("茶树 15 至 25 摄氏度")


def test_cached_embedder_avoids_recompute(tmp_path):
    class Counting(HashingEmbedder):
        def __init__(self):
            super().__init__(dim=64)
            self.calls = 0

        def encode(self, texts):
            self.calls += 1
            return super().encode(texts)

    inner = Counting()
    cache = CachedEmbedder(inner, path=tmp_path / "emb.json")
    first = cache.encode(["你好", "world"])
    assert inner.calls == 1
    second = cache.encode(["你好", "world"])
    assert inner.calls == 1, "命中缓存时不应再调用底层嵌入器"
    assert first == second
    assert (tmp_path / "emb.json").exists()
