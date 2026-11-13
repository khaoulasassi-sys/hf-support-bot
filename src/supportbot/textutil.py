"""Tokenisation et normalisation partagées (BM25, re-ranker à signaux, découpage de phrases)."""
from __future__ import annotations

import re
from functools import lru_cache

from nltk.stem import PorterStemmer

_stem = PorterStemmer().stem

STOPWORDS = set("""
a an the and or but if then else of to in on at by for with from into over under about as is are was were be been being
do does did done doing have has had having it its this that these those i me my we our you your he she they them their
what which who whom whose when where why how can could should would will shall may might must not no yes than too very
just also so such any all each both more most other some only own same up down out off again there here via using use
used get got let make made need want please tell show give help hi hello thanks thank
""".split())

_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


@lru_cache(maxsize=100_000)
def _stem_cached(tok: str) -> str:
    return _stem(tok)


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    """Tokens minuscules + racine. Les identifiants de code sont gardés entiers ET éclatés :
    `from_pretrained` → [from_pretrained, from, pretrain] ; `TrainingArguments` → [trainingarguments, train, argument]."""
    out: list[str] = []
    for raw in _TOKEN.findall(text):
        parts = [raw]
        if "_" in raw or (raw != raw.lower() and raw != raw.upper() and raw[1:] != raw[1:].lower()):
            sub = [p for chunk in raw.split("_") for p in _CAMEL.findall(chunk)]
            if len(sub) > 1:
                parts += sub
        for i, p in enumerate(parts):
            low = p.lower()
            if not low or (not keep_stopwords and low in STOPWORDS):
                continue
            is_whole_identifier = i == 0 and len(parts) > 1   # identifiant complet : gardé tel quel
            out.append(low if is_whole_identifier else _stem_cached(low))
    return out


def identifiers(text: str) -> list[str]:
    """Identifiants « techniques » d'une question : snake_case, CamelCase, noms avec point, options --flag."""
    ids = re.findall(r"\b[a-z]+(?:_[a-z0-9]+)+\b|\b[A-Z][a-z]+(?:[A-Z][a-z0-9]+)+\b|--[\w-]+|\b\w+\.\w+\(?", text)
    return [i.lower().strip("(") for i in ids]


def split_sentences(text: str) -> list[str]:
    """Découpage de phrases simple (suffisant pour la réponse extractive)."""
    text = re.sub(r"\s+", " ", text.strip())
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`\[(])", text) if s.strip()]
