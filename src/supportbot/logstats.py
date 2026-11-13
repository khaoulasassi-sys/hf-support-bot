"""Analyse des journaux de requêtes (logs/queries.jsonl) : usage, décisions, latence, questions sans réponse."""
from __future__ import annotations

import json
import statistics as st
from collections import Counter
from pathlib import Path


def summarize(path: Path) -> dict:
    if not path.exists():
        print(f"Aucun journal : {path}")
        return {}
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(rows)
    dec = Counter(r["decision"] for r in rows)
    lat = [r["latency_ms"].get("total", 0) for r in rows]
    s = {"n_queries": n, "decisions": dict(dec),
         "escalation_rate": round(sum(r["escalation_suggested"] for r in rows) / n, 3),
         "mean_confidence": round(st.mean(r["confidence"] for r in rows), 3),
         "latency_ms": {"mean": round(st.mean(lat), 1), "p95": round(sorted(lat)[int(0.95 * (n - 1))], 1)},
         "llm_errors": sum(bool(r.get("llm_error")) for r in rows),
         "least_confident": [(r["question"][:70], r["confidence"]) for r in sorted(rows, key=lambda r: r["confidence"])[:5]],
         "most_cited_pages": Counter(s["doc_id"] for r in rows for s in r["top_sources"][:1]).most_common(5)}
    print(json.dumps(s, indent=2, ensure_ascii=False))
    return s
