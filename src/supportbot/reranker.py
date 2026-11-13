"""Re-ranking : seconde étape, plus coûteuse mais plus précise que le retrieval initial.

Le retrieval (bi-encodeur + BM25) encode question et passage *séparément* : rapide, mais approximatif.
Un cross-encoder lit la paire (question, passage) *ensemble* : il capte les interactions fines
mot-à-mot et sépare bien mieux « parle du même sujet » de « répond à la question ».

Deux implémentations :
* CrossEncoderReranker (profil hf) : cross-encoder/ms-marco-MiniLM-L-6-v2 ;
* FeatureReranker (profil offline) : régression logistique apprise sur 8 signaux (couverture
  des termes, bigrammes, titres, identifiants de code, scores dense/BM25…).
Les deux renvoient une *probabilité de pertinence* ∈ [0,1], utilisée comme score de confiance.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

import numpy as np

from .textutil import identifiers, tokenize
from .types import Candidate


class Reranker(Protocol):
    def rerank(self, query: str, cands: list[Candidate], top_n: int) -> list[Candidate]: ...


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 30.0), -30.0)))


class NoReranker:
    """Pas de re-ranking (ablation) : le score = cosinus du bi-encodeur, borné à [0,1]."""

    def rerank(self, query, cands, top_n):
        for c in cands:
            c.rerank_score = float(min(max(c.dense_score, 0.0), 1.0))
        return sorted(cands, key=lambda c: c.fused_score, reverse=True)[:top_n]


class CrossEncoderReranker:
    def __init__(self, model: str, max_length: int = 512, batch_size: int = 16):
        import torch
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model, max_length=max_length)
        self.bs = batch_size
        self._identity = torch.nn.Identity()

    def rerank(self, query, cands, top_n):
        if not cands:
            return []
        pairs = [(query, c.chunk.embed_text) for c in cands]
        # logits bruts (activation = identité) puis sigmoïde : même échelle quel que soit le modèle
        logits = self.model.predict(pairs, batch_size=self.bs, activation_fn=self._identity,
                                    show_progress_bar=False)
        for c, s in zip(cands, np.asarray(logits).reshape(-1)):
            c.rerank_score = _sigmoid(float(s))
        return sorted(cands, key=lambda c: c.rerank_score, reverse=True)[:top_n]


# ---------------------------------------------------------------------------------------------
FEATURE_NAMES = ["dense", "dense_gap", "bm25_rel", "fused_rel", "coverage", "title_cov", "bigram", "ident"]


def _bigrams(toks: list[str]) -> set[tuple[str, str]]:
    return set(zip(toks, toks[1:]))


def extract_features(query: str, cands: list[Candidate]) -> np.ndarray:
    """Matrice (n_candidats × 8) de signaux de pertinence pour une même requête."""
    q_toks = list(dict.fromkeys(tokenize(query)))        # termes uniques, ordre conservé
    q_seq = tokenize(query)
    q_big = _bigrams(q_seq)
    q_ids = identifiers(query)
    max_dense = max((c.dense_score for c in cands), default=0.0)
    max_bm25 = max((c.bm25_score for c in cands), default=0.0) or 1.0
    max_fused = max((c.fused_score for c in cands), default=0.0) or 1.0
    rows = []
    for c in cands:
        p_toks = set(tokenize(c.chunk.text))
        h_toks = set(tokenize(c.chunk.doc_title + " " + c.chunk.section))
        p_seq = tokenize(c.chunk.text)
        cov = sum(t in p_toks or t in h_toks for t in q_toks) / len(q_toks) if q_toks else 0.0
        tcov = sum(t in h_toks for t in q_toks) / len(q_toks) if q_toks else 0.0
        big = len(q_big & _bigrams(p_seq)) / len(q_big) if q_big else 0.0
        low = c.chunk.text.lower()
        ident = sum(i in low for i in q_ids) / len(q_ids) if q_ids else 0.0
        rows.append([c.dense_score, c.dense_score - max_dense, c.bm25_score / max_bm25,
                     c.fused_score / max_fused, cov, tcov, big, ident])
    return np.asarray(rows, dtype="float32")


class FeatureReranker:
    def __init__(self, model_path: str | Path):
        import joblib

        path = Path(model_path)
        if not path.is_absolute():
            from .config import ROOT
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"{path} introuvable : lancez `python -m supportbot calibrate --profile offline` "
                                    "(entraîne le re-ranker sur le jeu de dev).")
        self.model = joblib.load(path)

    def rerank(self, query, cands, top_n):
        if not cands:
            return []
        probs = self.model.predict_proba(extract_features(query, cands))[:, 1]
        for c, p in zip(cands, probs):
            c.rerank_score = float(p)
        return sorted(cands, key=lambda c: c.rerank_score, reverse=True)[:top_n]


def train_feature_reranker(samples: list[tuple[str, list[Candidate], list[int]]], out_path: Path):
    """samples = [(question, candidats, labels 0/1)] → régression logistique standardisée."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([extract_features(q, cands) for q, cands, _ in samples])
    y = np.concatenate([lab for _, _, lab in samples])
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    model.fit(X, y)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    coefs = dict(zip(FEATURE_NAMES, model[-1].coef_[0].round(3).tolist()))
    return model, {"n_rows": int(len(y)), "n_pos": int(y.sum()), "coefficients": coefs}


def make_reranker(cfg: dict) -> Reranker:
    r = cfg["rerank"]
    if r["backend"] == "cross-encoder":
        return CrossEncoderReranker(r["model"], r.get("max_length", 512))
    if r["backend"] == "feature":
        return FeatureReranker(r["model"])
    if r["backend"] == "none":
        return NoReranker()
    raise ValueError(f"Backend de re-ranking inconnu : {r['backend']}")
