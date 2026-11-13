"""Score de confiance et décision (répondre / répondre avec réserve / refuser).

Le score de confiance est la probabilité de pertinence du meilleur passage après re-ranking.
Deux seuils découpent trois zones, calibrées sur un jeu de développement (voir calibrate) :

    confiance ≥ tau_high          → "confident"  : réponse normale
    tau_low ≤ confiance < tau_high → "uncertain"  : réponse avec avertissement + escalade proposée
    confiance < tau_low           → "no_answer"  : « je ne sais pas » + escalade proposée
"""
from __future__ import annotations

import re

from .types import Candidate

HUMAN_REQUEST = re.compile(
    r"\b(human|real person|human agent|support team|speak to (?:someone|a person)|talk to (?:someone|a person)|"
    r"humain|conseiller|parler à quelqu|contacter le support|escalate)\b", re.I)


def assess(reranked: list[Candidate], tau_high: float, tau_low: float) -> tuple[str, float]:
    conf = reranked[0].rerank_score if reranked and reranked[0].rerank_score is not None else 0.0
    if conf >= tau_high:
        return "confident", conf
    if conf >= tau_low:
        return "uncertain", conf
    return "no_answer", conf


def wants_human(question: str) -> bool:
    return bool(HUMAN_REQUEST.search(question))


def is_complex(question: str) -> str:
    """Heuristiques de complexité : l'escalade est suggérée même si le retrieval est confiant."""
    if question.count("?") >= 3:
        return "la question contient plusieurs demandes distinctes"
    if len(question.split()) > 150 or "Traceback (most recent call last)" in question:
        return "un diagnostic détaillé (trace d'erreur / contexte long) dépasse la documentation"
    return ""
