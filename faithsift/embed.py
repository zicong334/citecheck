# -*- coding: utf-8 -*-
"""可选的向量嵌入通道(可插拔、带缓存、永不强制)。

词法 BM25 引擎在精确措辞和数字上很强,但对重度同义改写、开放领域的
跨语言配对束手无策。本模块补上这条缺失的通道,同时不引入数值依赖:
向量就是普通的 ``list[float]``,余弦就是点积,不需要 numpy。

* :class:`Embedder` —— 只有 ``encode`` 一个方法的协议;
* :class:`OpenAIEmbedder` —— 任何 OpenAI 兼容的 ``/embeddings`` 端点
  (代理、本地 vLLM、Ollama 的 OpenAI 兼容层等);
* :class:`LocalOnnxEmbedder` —— **本地离线** ONNX 多语后端(fastembed,不含 torch),
  ``--embedder local`` 启用;惰性导入,缺依赖时抛 :class:`SemanticBackendUnavailable`
  让上层退回词法,核心包不装向量依赖也能 import;
* :class:`HashingEmbedder` —— 确定性、零网络的替身,供测试与
  ``--embedder hashing`` 使用;它把词/字符 n-gram 做带符号随机投影,
  能抹平词形变化,但**不能**跨语言;
* :class:`CachedEmbedder` —— 内容寻址的磁盘缓存,重复运行和 CI
  不会为同一段文本重复付费;
* :class:`DenseIndex` —— 片段矩阵 + 查询余弦打分。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Protocol, Sequence

DEFAULT_DIM = 384
_CACHE_VERSION = 1

# 本地离线向量后端(可选,装在 extras[semantic] 里):
# fastembed 基于 onnxruntime,不引入 torch,比 sentence-transformers 轻得多。
# 默认模型是 384 维的多语 MiniLM,能中→英跨语言,且**不需要** query/passage 前缀,
# 正好契合本模块"只有 encode(texts)"的 Embedder 协议(e5 系列需要前缀,故列为备选)。
DEFAULT_LOCAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_MIRROR = "https://hf-mirror.com"
_REPO_ROOT = Path(__file__).resolve().parent.parent


class SemanticBackendUnavailable(RuntimeError):
    """本地向量后端不可用(缺 fastembed 依赖、或模型加载/下载失败)。

    单独成一个类型,是为了让 :func:`faithsift.retrieval.build_retriever` 能精准
    捕获它、打印中文提示并**自动退回词法通道**,而不是让整条链路崩掉——
    与"没有 key 时 hybrid 退回 lexical"是同一套降级哲学。
    """


class Embedder(Protocol):
    model: str

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


def _norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def _l2(vec: list[float]) -> list[float]:
    n = _norm(vec)
    return [v / n for v in vec] if n > 1e-12 else vec


class HashingEmbedder:
    """确定性哈希嵌入器(不联网、不下载模型)。

    特征由单词一元组与字符三元组构成(英文词形变化、中文用词变化都会落在
    相近位置),哈希进 ``dim`` 个带符号桶。便宜且完全确定,适合测试与粗粒度
    模糊匹配,但它跨不了语言。
    """

    def __init__(self, dim: int = DEFAULT_DIM):
        self.dim = dim
        self.model = f"hashing-{dim}"

    @staticmethod
    def _features(text: str) -> list[str]:
        low = text.lower()
        words = re.findall(r"[a-z0-9]+", low)
        feats = list(words)
        for w in words:
            feats.extend(w[i:i + 3] for i in range(max(0, len(w) - 2)))
        cjk = re.findall(r"[一-鿿]", low)
        feats.extend(cjk)
        feats.extend(a + b for a, b in zip(cjk, cjk[1:]))
        feats.extend(a + b + c for a, b, c in zip(cjk, cjk[1:], cjk[2:]))
        return feats

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for f in self._features(text):
            h = int.from_bytes(
                hashlib.blake2b(f.encode("utf-8"), digest_size=8).digest(), "big")
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[h % self.dim] += sign
        return _l2(vec)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]


class OpenAIEmbedder:
    """任何 OpenAI 兼容的 embeddings 端点。"""

    def __init__(self, model: str = "text-embedding-3-small",
                 base_url: str | None = None, api_key: str | None = None,
                 batch: int = 64):
        from openai import OpenAI

        key = (api_key or os.environ.get("FAITHSIFT_EMBED_KEY")
               or os.environ.get("CITECHECK_EMBED_KEY")  # 旧名回退
               or os.environ.get("OPENAI_API_KEY"))
        if not key:
            raise RuntimeError(
                "no embedding key (FAITHSIFT_EMBED_KEY or OPENAI_API_KEY); "
                "use --retriever lexical or --embedder hashing instead")
        self.client = OpenAI(
            api_key=key,
            base_url=(base_url
                       or os.environ.get("FAITHSIFT_EMBED_BASE_URL")
                       or os.environ.get("CITECHECK_EMBED_BASE_URL")  # 旧名回退
                       or "https://api.openai.com/v1"),
            timeout=90, max_retries=1)
        self.model = model
        self.batch = batch

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            chunk = list(texts[i:i + self.batch])
            resp = self.client.embeddings.create(model=self.model, input=chunk)
            out.extend([list(d.embedding) for d in resp.data])
        return out


class LocalOnnxEmbedder:
    """本地离线 ONNX 多语嵌入器(fastembed 优先;不引入 torch)。

    * **依赖隔离**:fastembed 只在 ``__init__`` 里惰性导入。没装
      ``extras[semantic]`` 时抛 :class:`SemanticBackendUnavailable`,由
      :func:`faithsift.retrieval.build_retriever` 捕获并退回词法通道——
      核心包不装向量依赖也能 import、也能跑 heuristic 引擎。
    * **国内镜像**:默认把 ``HF_ENDPOINT`` 设为 ``https://hf-mirror.com``
      (仅当用户未自行设置),首次下载走镜像;模型缓存到本地后完全离线。
    * **协议一致**:`encode` 把 numpy 数组转成普通 ``list[float]``,
      与 :class:`HashingEmbedder`/:class:`OpenAIEmbedder` 完全同构,可直接套
      :class:`CachedEmbedder` 复用磁盘缓存(键含模型名,不同模型不串味)。

    模型名支持别名:``local`` = 默认多语 MiniLM;``local:<hf-model-id>``
    可指定任意 fastembed 支持的模型(如 ``local:intfloat/multilingual-e5-small``)。
    退路说明:fastembed 本身即基于 onnxruntime,已是"比 torch 轻"的那条路;
    若连 fastembed 都装不上,可退到裸 onnxruntime + 手动下载 onnx,但那条路
    需要自带分词与池化,复杂度更高,故此处以 fastembed 为唯一实现。
    """

    ALIASES = {
        "local": DEFAULT_LOCAL_MODEL,
        "multilingual-minilm": DEFAULT_LOCAL_MODEL,
        "paraphrase-multilingual-minilm-l12-v2": DEFAULT_LOCAL_MODEL,
        "multilingual-e5-small": "intfloat/multilingual-e5-small",
    }

    def __init__(self, model: str = "local", cache_dir: Path | str | None = None,
                 local_files_only: bool = False, threads: int | None = None):
        self.model = self._resolve(model)
        # 只在用户没设置时兜底为国内镜像,尊重已有环境变量。
        os.environ.setdefault("HF_ENDPOINT", HF_MIRROR)
        if cache_dir is None:
            cache_dir = os.environ.get("FASTEMBED_CACHE_PATH") \
                or (_REPO_ROOT / ".cache" / "fastembed")
        try:
            from fastembed import TextEmbedding
        except ImportError as e:  # 缺依赖 -> 明确中文提示 + 让上层降级
            raise SemanticBackendUnavailable(
                "本地向量后端需要 fastembed(基于 onnxruntime,不含 torch)。"
                "请安装:pip install 'faithsift[semantic]'(或 pip install fastembed);"
                "离线自检可改用 --embedder hashing 或 --retriever lexical。") from e
        try:
            self._te = TextEmbedding(
                self.model, cache_dir=str(cache_dir), threads=threads,
                local_files_only=local_files_only)
        except Exception as e:  # noqa: BLE001  下载/加载失败都要能降级
            raise SemanticBackendUnavailable(
                f"本地嵌入模型 '{self.model}' 加载失败: {str(e)[:160]};"
                "首次使用需联网从 HF 镜像下载,或改用 --embedder hashing。") from e

    @classmethod
    def _resolve(cls, model: str) -> str:
        m = (model or "local").strip()
        if m.startswith("local:"):
            m = m[len("local:"):]
        return cls.ALIASES.get(m.lower(), m)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        return [[float(x) for x in v] for v in self._te.embed(texts)]


class CachedEmbedder:
    """套在任意嵌入器前面的内容寻址缓存。

    键 = ``sha256(model + "\\x00" + text)``,值是 JSON 浮点数组,全部存在一个
    文件里。命中缓存就完全跳过网络调用——正是一次性付钱、之后可复现的关键。
    """

    def __init__(self, inner: Embedder,
                 path: Path | str = Path(".cache/embeddings.json")):
        self.inner = inner
        self.model = getattr(inner, "model", "unknown")
        self.path = Path(path)
        self._mem: dict[str, list[float]] = {}
        self._load()

    def _key(self, text: str) -> str:
        h = hashlib.sha256()
        h.update(f"{_CACHE_VERSION}\x00{self.model}\x00{text}".encode("utf-8"))
        return h.hexdigest()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._mem = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._mem = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._mem), encoding="utf-8")
        tmp.replace(self.path)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self._mem]
        if missing:
            vecs = self.inner.encode([t for _, t in missing])
            for (k, _), v in zip(missing, vecs):
                self._mem[k] = list(v)
            self._save()
        return [self._mem[k] for k in keys]


class DenseIndex:
    """片段向量矩阵 + 查询余弦打分。"""

    def __init__(self, chunks: Sequence, embedder: Embedder):
        self.chunks = list(chunks)
        self.embedder = embedder
        vectors = embedder.encode([c.text for c in self.chunks]) \
            if self.chunks else []
        self.matrix = [_l2(list(v)) for v in vectors]
        self._q: dict[str, list[float]] = {}

    def _query(self, text: str) -> list[float]:
        if text not in self._q:
            self._q[text] = _l2(list(self.embedder.encode([text])[0]))
        return self._q[text]

    def cosine(self, query: str) -> list[float]:
        if not self.chunks:
            return []
        q = self._query(query)
        return [float(sum(a * b for a, b in zip(q, vec))) for vec in self.matrix]

    def stats(self) -> dict:
        dim = len(self.matrix[0]) if self.matrix else 0
        return {"model": getattr(self.embedder, "model", "unknown"),
                "n_chunks": len(self.chunks), "dim": dim,
                "finite": all(math.isfinite(x) for vec in self.matrix for x in vec)}


__all__ = ["Embedder", "HashingEmbedder", "OpenAIEmbedder", "LocalOnnxEmbedder",
           "CachedEmbedder", "DenseIndex", "SemanticBackendUnavailable",
           "DEFAULT_DIM", "DEFAULT_LOCAL_MODEL", "HF_MIRROR"]
