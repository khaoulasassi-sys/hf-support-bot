"""Modèles d'embeddings : un seul contrat, deux implémentations.

* SentenceTransformerEmbedder : n'importe quel modèle sentence-transformers / HuggingFace
  (E5, BGE, MiniLM…). Gère les préfixes « query: » / « passage: » exigés par E5.
* WordLlamaEmbedder : embeddings statiques fournis DANS le paquet pip `wordllama` (aucun
  téléchargement) — utilisé par le profil « offline ».
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol

import numpy as np


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-9, None)


class Embedder(Protocol):
    name: str
    dim: int

    def encode_passages(self, texts: list[str]) -> np.ndarray: ...
    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model: str, query_prefix: str = "", passage_prefix: str = "", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.name = model
        self.model = SentenceTransformer(model)
        self.qp, self.pp, self.bs = query_prefix, passage_prefix, batch_size
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def _enc(self, texts: list[str], prefix: str) -> np.ndarray:
        vecs = self.model.encode([prefix + t for t in texts], batch_size=self.bs,
                                 show_progress_bar=len(texts) > 200, convert_to_numpy=True)
        return _l2(vecs)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._enc(texts, self.pp)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._enc(texts, self.qp)


class WordLlamaEmbedder:
    def __init__(self, model: str = "l2_supercat_256", **_):
        import wordllama
        from wordllama import WordLlama

        # Contournement d'un défaut de wordllama 0.4 : le tokenizer est livré dans `tokenizers/`
        # mais cherché dans le cache sous le même nom ; on le copie pour éviter tout téléchargement.
        src = Path(wordllama.__file__).parent / "tokenizers" / "l2_supercat_tokenizer_config.json"
        cache = WordLlama.get_file_path("tokenizer", None)
        cache.mkdir(parents=True, exist_ok=True)
        if src.exists() and not (cache / src.name).exists():
            shutil.copy(src, cache / src.name)
        self.name = model
        self.model = WordLlama.load(disable_download=True)
        self.dim = 256

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return _l2(self.model.embed(texts))

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return _l2(self.model.embed(texts))


def make_embedder(cfg: dict) -> Embedder:
    e = cfg["embedding"]
    if e["backend"] == "sentence-transformers":
        return SentenceTransformerEmbedder(e["model"], e.get("query_prefix", ""),
                                           e.get("passage_prefix", ""), e.get("batch_size", 32))
    if e["backend"] == "wordllama":
        return WordLlamaEmbedder(e["model"])
    raise ValueError(f"Backend d'embedding inconnu : {e['backend']}")
