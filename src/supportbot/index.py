"""Index hybride : FAISS (dense) + BM25 (lexical), fusionnés par Reciprocal Rank Fusion (RRF).

Pourquoi hybride ? Dans une documentation technique, les questions contiennent souvent des
identifiants exacts (`from_pretrained`, `load_in_8bit`, `--fp16`) que le lexical retrouve mieux,
tandis que le dense gère les reformulations (« mon modèle est trop gros pour ma carte graphique »).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .embeddings import Embedder
from .textutil import tokenize
from .types import Candidate, Chunk


class HybridIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder):
        self.chunks = chunks
        self.vectors = vectors.astype("float32")
        self.embedder = embedder
        self.faiss = faiss.IndexFlatIP(self.vectors.shape[1])   # produit scalaire = cosinus (vecteurs normalisés)
        self.faiss.add(self.vectors)
        self.bm25 = BM25Okapi([tokenize(c.embed_text) for c in chunks])

    # ------------------------------------------------------------------ construction / persistance
    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> "HybridIndex":
        vectors = embedder.encode_passages([c.embed_text for c in chunks])
        return cls(chunks, vectors, embedder)

    def save(self, directory: Path, cfg: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        with open(directory / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        faiss.write_index(self.faiss, str(directory / "faiss.index"))
        meta = {"embedding_backend": cfg["embedding"]["backend"], "embedding_model": self.embedder.name,
                "dim": int(self.vectors.shape[1]), "n_chunks": len(self.chunks),
                "n_docs": len({c.doc_id for c in self.chunks}), "chunking": cfg["chunking"],
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "corpus_sha256": hashlib.sha256("".join(c.text for c in self.chunks).encode()).hexdigest()}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path, embedder: Embedder) -> "HybridIndex":
        if not (directory / "meta.json").exists():
            raise FileNotFoundError(f"Index introuvable dans {directory}. Lancez : python -m supportbot build")
        meta = json.loads((directory / "meta.json").read_text())
        if meta["embedding_model"] != embedder.name:
            raise ValueError(f"Index construit avec {meta['embedding_model']} mais config = {embedder.name} : "
                             "reconstruisez l'index (python -m supportbot build).")
        chunks = [Chunk(**json.loads(l)) for l in (directory / "chunks.jsonl").read_text().splitlines()]
        return cls(chunks, np.load(directory / "vectors.npy"), embedder)

    # ------------------------------------------------------------------ recherche
    def scores(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        """Scores denses (cosinus) et BM25 de TOUS les passages pour une requête."""
        qv = self.embedder.encode_queries([query])[0]
        return self.vectors @ qv, np.asarray(self.bm25.get_scores(tokenize(query)), dtype="float32")

    def search(self, query: str, mode: str = "hybrid", k_dense: int = 30, k_bm25: int = 30,
               k_candidates: int = 20, rrf_k: int = 60) -> list[Candidate]:
        dense, bm25 = self.scores(query)
        d_rank = np.argsort(-dense)[:k_dense]
        b_rank = np.argsort(-bm25)[:k_bm25]
        fused: dict[int, float] = {}
        if mode in ("dense", "hybrid"):
            for r, i in enumerate(d_rank):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (rrf_k + r + 1)
        if mode in ("bm25", "hybrid"):
            for r, i in enumerate(b_rank):
                if bm25[i] > 0:
                    fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (rrf_k + r + 1)
        top = sorted(fused, key=fused.get, reverse=True)[:k_candidates]
        return [Candidate(self.chunks[i], float(dense[i]), float(bm25[i]), fused[i]) for i in top]
