"""Console de support — interface Streamlit du chatbot (historique, sources, confiance, escalade).

Lancer :  streamlit run app/streamlit_app.py
Profil / LLM modifiables dans la barre latérale ou via SUPPORTBOT_PROFILE, SUPPORTBOT_LLM.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from supportbot.config import load_config, resolve  # noqa: E402
from supportbot.pipeline import SupportBot  # noqa: E402

st.set_page_config(page_title="Support Transformers", page_icon="🛟", layout="wide")

BADGE = {"confident": ("🟢", "Confiant"), "uncertain": ("🟠", "Incertain"),
         "no_answer": ("🔴", "Pas de réponse fiable"), "human_requested": ("🙋", "Expert humain demandé")}


@st.cache_resource(show_spinner="Chargement de l'index et des modèles…")
def get_bot(profile: str, llm_backend: str, llm_model: str, base_url: str) -> SupportBot:
    overrides = {"llm": {"backend": llm_backend}}
    if llm_backend == "openai":
        overrides["llm"].update({"model": llm_model, "base_url": base_url})
    if llm_backend == "hf-local":
        overrides["llm"]["hf_model"] = llm_model
    return SupportBot.from_config(profile, overrides)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ barre latérale
with st.sidebar:
    st.header("⚙️ Configuration")
    profile = st.selectbox("Profil", ["hf", "offline"], index=["hf", "offline"].index(os.environ.get("SUPPORTBOT_PROFILE", "hf")),
                           help="hf : E5 + cross-encoder (téléchargés). offline : sans réseau.")
    default_llm = os.environ.get("SUPPORTBOT_LLM", "extractive")
    llm_backend = st.selectbox("Générateur de réponse", ["extractive", "openai", "hf-local"],
                               index=["extractive", "openai", "hf-local"].index(default_llm),
                               help="extractive : sans LLM. openai : Ollama / OpenAI / Groq… hf-local : modèle transformers.")
    cfg0 = load_config(profile)
    llm_model = st.text_input("Modèle LLM", cfg0["llm"]["model"] if llm_backend == "openai" else cfg0["llm"]["hf_model"],
                              disabled=llm_backend == "extractive")
    base_url = st.text_input("URL API (compatible OpenAI)", cfg0["llm"]["base_url"], disabled=llm_backend != "openai")
    try:
        bot = get_bot(profile, llm_backend, llm_model, base_url)
    except Exception as e:  # index absent, modèle non téléchargé…
        st.error(f"Impossible de démarrer le bot : {e}")
        st.stop()
    c = bot.cfg["confidence"]
    st.caption(f"Index : {len(bot.index.chunks)} passages · embeddings `{bot.index.embedder.name}`")
    st.caption(f"Seuils de confiance : τ_high = {c['tau_high']:.2f} · τ_low = {c['tau_low']:.2f}")
    if st.button("🗑️ Nouvelle conversation"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.subheader("🎫 Tickets d'escalade")
    esc = resolve(bot.cfg, "log_dir") / bot.cfg["logging"]["escalations_file"]
    tickets = [json.loads(l) for l in esc.read_text().splitlines()] if esc.exists() else []
    st.write(f"{len(tickets)} ticket(s) ouvert(s)")
    for t in tickets[-5:][::-1]:
        st.caption(f"**{t['ticket_id']}** — {t['question'][:60]}")

# ------------------------------------------------------------------ conversation
st.title("🛟 Support Hugging Face Transformers")
st.caption("Assistant technique basé sur la documentation officielle · réponses sourcées · escalade vers un expert humain")

if "messages" not in st.session_state:
    st.session_state.messages = []


def history_pairs() -> list[tuple[str, str]]:
    msgs, pairs = st.session_state.messages, []
    for i in range(0, len(msgs) - 1, 2):
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
            pairs.append((msgs[i]["content"], msgs[i + 1]["content"]))
    return pairs


def render_bot(m: dict, idx: int) -> None:
    icon, label = BADGE[m["decision"]]
    st.markdown(m["content"])
    st.caption(f"{icon} **{label}** · confiance {m['confidence']:.0%} · {m['latency'].get('total', 0):.0f} ms")
    if m["sources"]:
        with st.expander(f"📚 Passages utilisés ({len(m['sources'])})"):
            for i, s in enumerate(m["sources"], 1):
                st.markdown(f"**[{i}] [{s['title']}]({s['url']})**  \n"
                            f"`dense {s['dense']:.2f}` · `bm25 {s['bm25']:.1f}` · `re-ranking {s['rerank']:.2f}`")
                st.text(s["text"][:700] + ("…" if len(s["text"]) > 700 else ""))
    cols = st.columns([1, 1, 2, 5])
    if cols[0].button("👍", key=f"up{idx}"):
        append_jsonl(resolve(bot.cfg, "log_dir") / "feedback.jsonl", {"ts": datetime.now(timezone.utc).isoformat(), "q": m["question"], "vote": 1})
        st.toast("Merci pour votre retour !")
    if cols[1].button("👎", key=f"down{idx}"):
        append_jsonl(resolve(bot.cfg, "log_dir") / "feedback.jsonl", {"ts": datetime.now(timezone.utc).isoformat(), "q": m["question"], "vote": -1})
        st.toast("Retour enregistré.")
    if m["escalation_suggested"]:
        if m.get("ticket"):
            cols[3].success(f"Ticket **{m['ticket']}** créé — un expert vous répondra.")
        elif cols[2].button("🙋 Escalader", key=f"esc{idx}", help=f"Raison : {m['escalation_reason']}"):
            hist = history_pairs()[:-1]
            t = bot.escalate(m["question"], hist, m["escalation_reason"] or "demande de l'utilisateur")
            m["ticket"] = t["ticket_id"]
            st.rerun()


for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            render_bot(m, i)
        else:
            st.markdown(m["content"])

if prompt := st.chat_input("Posez votre question technique (ex. « How do I load a model in 4-bit? »)"):
    hist = history_pairs()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Recherche dans la documentation…"):
            r = bot.answer(prompt, hist)
        meta = {"role": "assistant", "content": r.answer, "decision": r.decision, "confidence": r.confidence,
                "latency": r.latency_ms, "question": prompt, "escalation_suggested": r.escalation_suggested,
                "escalation_reason": r.escalation_reason, "ticket": None,
                "sources": [{"title": s.chunk.doc_title + (f" > {s.chunk.section}" if s.chunk.section else ""),
                             "url": s.chunk.url, "dense": s.dense_score, "bm25": s.bm25_score,
                             "rerank": s.rerank_score or 0.0, "text": s.chunk.text} for s in r.sources]}
        if r.decision == "human_requested":                    # demande explicite : ticket immédiat
            meta["ticket"] = bot.escalate(prompt, hist, r.escalation_reason, r)["ticket_id"]
        st.session_state.messages.append(meta)
    st.rerun()
