"""Pipeline complet du chatbot de support : retrieval → re-ranking → confiance → réponse / escalade.

    question ─► (reformulation avec l'historique) ─► index hybride (dense + BM25, RRF)
             ─► re-ranker (cross-encoder) ─► score de confiance ─► décision
                   ├─ confident  : réponse du LLM + sources
                   ├─ uncertain  : réponse avec réserve + escalade proposée
                   └─ no_answer  : « je ne sais pas » + escalade proposée (le LLM n'est PAS appelé)
"""
from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import confidence as conf
from .config import load_config, resolve
from .embeddings import make_embedder
from .index import HybridIndex
from .llm import ExtractiveLLM, LLMError, looks_french, make_llm
from .reranker import make_reranker
from .types import BotResponse, Candidate

MESSAGES = {
    "no_answer": {
        "en": "I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I "
              "would rather not guess. I can pass your question to a human expert.",
        "fr": "Je n'ai pas trouvé de réponse fiable à cette question dans la documentation de Hugging Face "
              "Transformers, et je préfère ne pas deviner. Je peux transmettre votre question à un expert humain.",
    },
    "uncertain_footer": {
        "en": "⚠️ I'm only moderately confident ({c:.0%}). Please check the sources, or ask for a human expert.",
        "fr": "⚠️ Ma confiance est moyenne ({c:.0%}). Vérifiez les sources, ou demandez un expert humain.",
    },
    "human": {
        "en": "Understood — I'm creating a ticket for a human expert, with a summary of our conversation.",
        "fr": "C'est noté — je crée un ticket pour un expert humain, avec le résumé de notre conversation.",
    },
    "complex": {
        "en": "Note: {why}. A human expert may be better placed to help.",
        "fr": "Remarque : {why}. Un expert humain sera peut-être mieux placé pour vous aider.",
    },
    "llm_down": {
        "en": "(The language model is unavailable — showing an extract of the documentation instead.)",
        "fr": "(Le modèle de langage est indisponible — voici un extrait de la documentation à la place.)",
    },
}

_ANAPHORA = re.compile(r"^(and|also|what about|how about|et|et pour|et si|what if|ok|then)\b|"
                       r"\b(it|that|this|them|those|ça|cela|celui|celle)\b", re.I)


class SupportBot:
    def __init__(self, cfg: dict, index: HybridIndex, reranker, llm):
        self.cfg, self.index, self.reranker, self.llm = cfg, index, reranker, llm
        self.fallback_llm = ExtractiveLLM()
        self.log_dir = resolve(cfg, "log_dir")

    @classmethod
    def from_config(cls, profile_or_cfg: str | dict = "hf", overrides: dict | None = None) -> "SupportBot":
        cfg = load_config(profile_or_cfg, overrides) if isinstance(profile_or_cfg, str) else profile_or_cfg
        embedder = make_embedder(cfg)
        index = HybridIndex.load(resolve(cfg, "index_dir"), embedder)
        return cls(cfg, index, make_reranker(cfg), make_llm(cfg))

    # ------------------------------------------------------------------ étapes
    def _contextualize(self, question: str, history: list[tuple[str, str]]) -> str:
        """Question de suivi (« et sur GPU ? ») → on la complète avec la question précédente."""
        if history and (len(question.split()) <= 5 or _ANAPHORA.search(question)):
            return f"{history[-1][0]} {question}"
        return question

    def retrieve(self, query: str) -> list[Candidate]:
        r = self.cfg["retrieval"]
        return self.index.search(query, r["mode"], r["k_dense"], r["k_bm25"], r["k_candidates"], r["rrf_k"])

    def answer(self, question: str, history: list[tuple[str, str]] | None = None, log: bool = True) -> BotResponse:
        history = (history or [])[-self.cfg["history"]["max_turns"]:]
        lang = "fr" if looks_french(question) else "en"
        t0 = time.perf_counter()

        if conf.wants_human(question):
            resp = BotResponse(question, MESSAGES["human"][lang], "human_requested", 1.0,
                               escalation_suggested=True, escalation_reason="demande explicite de l'utilisateur",
                               retrieval_query=question, latency_ms={"total": _ms(t0)})
            if log:
                self._log(resp, [], None)
            return resp

        rq = self._contextualize(question, history)
        t = time.perf_counter()
        cands = self.retrieve(rq)
        lat = {"retrieve": _ms(t)}
        t = time.perf_counter()
        top = self.reranker.rerank(rq, cands, self.cfg["rerank"]["top_n"])
        lat["rerank"] = _ms(t)
        c = self.cfg["confidence"]
        decision, score = conf.assess(top, c["tau_high"], c["tau_low"])
        complexity = conf.is_complex(question)

        llm_error = None
        if decision == "no_answer" or not top:
            answer, lat["generate"] = MESSAGES["no_answer"][lang], 0.0
            sources: list[Candidate] = []
        else:
            t = time.perf_counter()
            try:
                body = self.llm.generate(question, top, history, low_confidence=decision == "uncertain")
            except LLMError as e:                       # robustesse : repli extractif si le LLM est indisponible
                llm_error = str(e)
                body = MESSAGES["llm_down"][lang] + "\n\n" + self.fallback_llm.generate(
                    question, top, history, low_confidence=decision == "uncertain")
            lat["generate"] = _ms(t)
            sources = top
            answer = body + "\n\n" + self.format_sources(top)
            if decision == "uncertain":
                answer += "\n\n" + MESSAGES["uncertain_footer"][lang].format(c=score)
        if complexity:
            answer += "\n\n" + MESSAGES["complex"][lang].format(why=complexity)

        reason = {"no_answer": "confiance insuffisante", "uncertain": "confiance moyenne"}.get(decision, complexity)
        resp = BotResponse(question, answer, decision, score, sources,
                           escalation_suggested=decision != "confident" or bool(complexity),
                           escalation_reason=reason or "", retrieval_query=rq,
                           latency_ms={**lat, "total": _ms(t0)})
        if log:
            self._log(resp, cands, llm_error)
        return resp

    @staticmethod
    def format_sources(passages: list[Candidate]) -> str:
        lines = ["**Sources**"]
        for i, p in enumerate(passages, 1):
            title = p.chunk.doc_title + (f" > {p.chunk.section}" if p.chunk.section else "")
            lines.append(f"[{i}] {title} — {p.chunk.url}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ escalade & logs
    def escalate(self, question: str, history: list[tuple[str, str]], reason: str,
                 response: BotResponse | None = None) -> dict:
        """Crée un ticket d'escalade (fichier JSONL) avec tout le contexte utile à l'expert humain."""
        ticket = {
            "ticket_id": "T-" + uuid.uuid4().hex[:8].upper(),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "open", "question": question, "reason": reason,
            "conversation": [{"user": u, "bot": a[:500]} for u, a in history[-5:]],
            "bot_confidence": None if response is None else round(response.confidence, 3),
            "bot_decision": None if response is None else response.decision,
            "closest_sources": [] if response is None else [s.chunk.url for s in response.sources],
        }
        self._append(self.cfg["logging"]["escalations_file"], ticket)
        return ticket

    def _log(self, resp: BotResponse, cands: list[Candidate], llm_error: str | None) -> None:
        self._append(self.cfg["logging"]["queries_file"], {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "profile": self.cfg["profile"], "question": resp.question, "retrieval_query": resp.retrieval_query,
            "decision": resp.decision, "confidence": round(resp.confidence, 4),
            "escalation_suggested": resp.escalation_suggested, "latency_ms": resp.latency_ms,
            "llm": getattr(self.llm, "name", "?"), "llm_error": llm_error,
            "top_sources": [s.to_log() for s in resp.sources],
            "candidates": [c.to_log() for c in cands[:10]],
        })

    def _append(self, filename: str, record: dict) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_dir / filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
