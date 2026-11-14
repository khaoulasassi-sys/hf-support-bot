#!/usr/bin/env python3
"""Vérifie les étiquettes du jeu d'évaluation contre le corpus : chaque question in_scope doit avoir
(1) des pages cibles existantes, (2) au moins un passage de ces pages contenant un mot-clé."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
chunks = [json.loads(l) for l in (ROOT / "data/processed/chunks.jsonl").read_text(encoding="utf-8").splitlines()]
by_doc = {}
for c in chunks:
    by_doc.setdefault(c["doc_id"], []).append(c)
bad = 0
for l in (ROOT / "data/eval/questions.jsonl").read_text(encoding="utf-8").splitlines():
    q = json.loads(l)
    if q["category"] != "in_scope":
        continue
    missing = [d for d in q["gold_docs"] if d not in by_doc]
    hits = sum(any(k.lower() in c["text"].lower() for k in q["keywords"]) for d in q["gold_docs"] for c in by_doc.get(d, []))
    if missing or hits == 0:
        bad += 1
        print("PROBLÈME", q["id"], q["question"][:60], "| pages absentes:", missing, "| passages pertinents:", hits)
    elif hits == 1:
        print("fragile (1 seul passage pertinent)", q["id"], q["question"][:60])
print("OK" if not bad else f"{bad} question(s) à corriger")
sys.exit(1 if bad else 0)
