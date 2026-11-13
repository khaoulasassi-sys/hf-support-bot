import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from supportbot import confidence as conf
from supportbot.config import load_config
from supportbot.index import HybridIndex
from supportbot.llm import ExtractiveLLM, LLMError, OpenAICompatibleLLM, build_user_message, looks_french
from supportbot.pipeline import SupportBot
from supportbot.reranker import FeatureReranker, NoReranker, extract_features, train_feature_reranker
from supportbot.textutil import tokenize
from supportbot.types import Candidate


def test_index_search_roundtrip_and_model_guard(chunks, fake_embedder, tmp_path):
    idx = HybridIndex.build(chunks, fake_embedder)
    top = idx.search("how to use gradient accumulation steps", k_candidates=3)
    assert top[0].chunk.doc_id == "training" and "gradient" in top[0].chunk.text
    assert idx.search("load_in_4bit", mode="bm25", k_candidates=2)[0].chunk.doc_id == "quant"
    cfg = load_config("offline")
    idx.save(tmp_path, cfg)
    again = HybridIndex.load(tmp_path, fake_embedder)
    assert [c.chunk.chunk_id for c in again.search("offline mode", k_candidates=3)] == \
           [c.chunk.chunk_id for c in idx.search("offline mode", k_candidates=3)]
    fake_embedder.name = "other-model"
    with pytest.raises(ValueError):
        HybridIndex.load(tmp_path, fake_embedder)
    fake_embedder.name = "fake-bow"


def test_confidence_zones_and_flags():
    mk = lambda s: [Candidate(None, rerank_score=s)]
    assert conf.assess(mk(0.9), 0.5, 0.1) == ("confident", 0.9)
    assert conf.assess(mk(0.3), 0.5, 0.1)[0] == "uncertain"
    assert conf.assess(mk(0.01), 0.5, 0.1)[0] == "no_answer"
    assert conf.assess([], 0.5, 0.1) == ("no_answer", 0.0)
    assert conf.wants_human("Can I speak to a human please?") and conf.wants_human("Je veux un conseiller humain")
    assert not conf.wants_human("How do I use agents with tools?")
    assert conf.is_complex("a? b? c?") and conf.is_complex("Traceback (most recent call last):\n x")
    assert not conf.is_complex("How do I install it?")


def test_feature_reranker_trains_and_orders(chunks, fake_embedder, tmp_path):
    idx = HybridIndex.build(chunks, fake_embedder)
    q = "gradient accumulation steps for a larger batch"
    cands = idx.search(q, k_candidates=len(chunks))
    assert extract_features(q, cands).shape == (len(cands), 8)
    labels = [int(c.chunk.doc_id == "training" and "gradient" in c.chunk.text) for c in cands]
    model, info = train_feature_reranker([(q, cands, labels)] * 3, tmp_path / "m.joblib")
    rr = FeatureReranker(tmp_path / "m.joblib")
    top = rr.rerank(q, idx.search(q, k_candidates=len(chunks)), 2)
    assert top[0].chunk.doc_id == "training" and 0 <= top[0].rerank_score <= 1 and info["n_pos"] >= 3


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.last = body
        if body["model"] == "boom":
            self.send_response(500); self.end_headers(); return
        out = json.dumps({"choices": [{"message": {"content": "Set `gradient_accumulation_steps` [1]."}}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture()
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


class LexicalStub:
    """Re-ranker de test : part des termes de la question présents dans le passage."""
    def rerank(self, query, cands, top_n):
        q = set(tokenize(query))
        for c in cands:
            c.rerank_score = len(q & set(tokenize(c.chunk.text + " " + c.chunk.section))) / max(len(q), 1)
        return sorted(cands, key=lambda c: c.rerank_score, reverse=True)[:top_n]


def _bot(chunks, fake_embedder, tmp_path, llm=None, tau=(0.5, 0.1), reranker=None):
    cfg = load_config("offline", {"paths": {"log_dir": str(tmp_path)}, "confidence": {"tau_high": tau[0], "tau_low": tau[1]},
                                   "rerank": {"top_n": 3}})
    return SupportBot(cfg, HybridIndex.build(chunks, fake_embedder), reranker or NoReranker(), llm or ExtractiveLLM())


def test_openai_compatible_llm_and_prompt(chunks, fake_embedder, server):
    idx = HybridIndex.build(chunks, fake_embedder)
    top = idx.search("gradient accumulation", k_candidates=2)
    llm = OpenAICompatibleLLM("m", server)
    assert "gradient_accumulation_steps" in llm.generate("how?", top, [("hi", "hello")], low_confidence=True)
    sent = _Handler.last["messages"]
    assert sent[0]["role"] == "system" and "NEVER invent" in sent[0]["content"]
    assert "[1]" in sent[1]["content"] and "retrieval confidence is LOW" in sent[1]["content"] and "User: hi" in sent[1]["content"]
    with pytest.raises(LLMError):
        OpenAICompatibleLLM("boom", server).generate("q", top)
    with pytest.raises(LLMError):
        OpenAICompatibleLLM("m", "http://127.0.0.1:9/v1", timeout_s=1).generate("q", top)


def test_pipeline_decisions_logging_escalation(chunks, fake_embedder, tmp_path):
    bot = _bot(chunks, fake_embedder, tmp_path, tau=(0.6, 0.2), reranker=LexicalStub())
    r = bot.answer("How do I use gradient accumulation steps?")
    assert r.decision == "confident" and "**Sources**" in r.answer and r.sources and not r.escalation_suggested
    r2 = bot.answer("zzz qqq unrelated sourdough bread recipe")
    assert r2.decision == "no_answer" and r2.sources == [] and r2.escalation_suggested
    r3 = bot.answer("Can I talk to a human?")
    assert r3.decision == "human_requested" and r3.escalation_suggested
    t = bot.escalate(r2.question, [("q", "a")], r2.escalation_reason, r2)
    assert t["ticket_id"].startswith("T-") and t["bot_decision"] == "no_answer"
    log = [json.loads(l) for l in (tmp_path / "queries.jsonl").read_text().splitlines()]
    assert [x["decision"] for x in log] == ["confident", "no_answer", "human_requested"]
    assert "retrieve" in log[0]["latency_ms"] and log[0]["top_sources"]
    assert (tmp_path / "escalations.jsonl").exists()
    assert bot.answer("anything", log=False) and len(log) == 3


def test_followup_question_uses_history(chunks, fake_embedder, tmp_path):
    bot = _bot(chunks, fake_embedder, tmp_path)
    hist = [("How do I enable gradient accumulation?", "answer")]
    assert bot._contextualize("and on GPU?", hist).startswith("How do I enable gradient accumulation?")
    assert bot._contextualize("How do I install the library from source in a clean environment?", hist).startswith("How do I install")
    assert bot._contextualize("and on GPU?", []) == "and on GPU?"


def test_llm_failure_falls_back_to_extractive(chunks, fake_embedder, tmp_path, server):
    llm = OpenAICompatibleLLM("boom", server)
    bot = _bot(chunks, fake_embedder, tmp_path, llm=llm, tau=(0.0, 0.0), reranker=LexicalStub())
    r = bot.answer("gradient accumulation steps")
    assert "unavailable" in r.answer and "**Sources**" in r.answer
    assert json.loads((tmp_path / "queries.jsonl").read_text().splitlines()[-1])["llm_error"]


def test_language_and_extractive_format(chunks, fake_embedder):
    assert looks_french("Comment charger un modèle ?") and not looks_french("How do I load a model?")
    top = HybridIndex.build(chunks, fake_embedder).search("gradient accumulation steps", k_candidates=3)
    out = ExtractiveLLM().generate("gradient accumulation steps", top)
    assert "[1]" in out and "gradient_accumulation_steps" in out
    assert "build_user_message" and build_user_message("q", top, [], False).startswith("Documentation excerpts:")
