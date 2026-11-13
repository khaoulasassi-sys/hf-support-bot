"""Structures de données partagées par tous les modules."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Document:
    doc_id: str          # identifiant stable (ex. "quantization/bitsandbytes")
    title: str
    text: str            # Markdown nettoyé
    url: str             # URL publique (pour citer la source)
    fmt: str = "md"      # md | html | pdf | txt


@dataclass
class Chunk:
    chunk_id: str        # "<doc_id>#<n>"
    doc_id: str
    doc_title: str
    section: str         # chemin de titres : "Quantization > bitsandbytes > 8-bit"
    url: str             # URL avec ancre de section
    text: str            # contenu du passage
    n_words: int

    @property
    def embed_text(self) -> str:
        """Texte réellement encodé : titre de page + section + contenu."""
        return f"{self.doc_title} > {self.section}\n{self.text}" if self.section else f"{self.doc_title}\n{self.text}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Candidate:
    """Un passage récupéré, avec les scores des différentes étapes."""
    chunk: Chunk
    dense_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float | None = None     # probabilité de pertinence ∈ [0, 1] après re-ranking

    def to_log(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id, "doc_id": self.chunk.doc_id,
            "dense": round(self.dense_score, 4), "bm25": round(self.bm25_score, 4),
            "fused": round(self.fused_score, 5),
            "rerank": None if self.rerank_score is None else round(self.rerank_score, 4),
        }


@dataclass
class BotResponse:
    question: str
    answer: str
    decision: str                       # confident | uncertain | no_answer | human_requested
    confidence: float
    sources: list[Candidate] = field(default_factory=list)
    escalation_suggested: bool = False
    escalation_reason: str = ""
    retrieval_query: str = ""
    latency_ms: dict[str, float] = field(default_factory=dict)
