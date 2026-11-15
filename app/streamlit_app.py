"""Lumen — assistant de support Hugging Face Transformers (interface Streamlit).

Historique de conversation, jauge de confiance, cartes de sources avec scores, escalade vers un
expert humain, retours utilisateur. Lancer :  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import html
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

st.set_page_config(page_title="Lumen · Support Transformers", page_icon="✨", layout="wide")

# ----------------------------------------------------------------------------- identité visuelle
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--ink:#1F1B2E;--paper:#FBF7F0;--sand:#F2EBDF;--violet:#5B3DF5;--violet-soft:#ECE8FF;--saffron:#F5A623;
      --teal:#1F9D8A;--coral:#E4572E;--muted:#7A7390;--line:#E4DACB;}
html,body,[class*="css"]{font-family:'Inter',system-ui,sans-serif;color:var(--ink);}
.stApp{background:radial-gradient(1100px 500px at 85% -10%,#ECE8FF 0%,rgba(236,232,255,0) 60%),
       radial-gradient(900px 420px at -10% 0%,#FFF0D6 0%,rgba(255,240,214,0) 55%),var(--paper);}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1.6rem;max-width:1100px;}
h1,h2,h3{font-family:'Fraunces',Georgia,serif!important;letter-spacing:-.01em;}
code,pre{font-family:'JetBrains Mono',ui-monospace,monospace!important;}

/* ---- hero */
.hero{display:flex;align-items:center;gap:18px;margin:4px 0 6px;}
.logo{width:60px;height:60px;border-radius:18px;flex:none;display:grid;place-items:center;
      background:linear-gradient(135deg,#5B3DF5 0%,#8A5CFF 55%,#F5A623 120%);
      box-shadow:0 10px 26px -8px rgba(91,61,245,.55);}
.hero h1{margin:0;font-size:2.35rem;line-height:1;}
.hero .tag{color:var(--muted);margin-top:6px;font-size:.98rem;}
.pills{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0 18px;}
.pill{font-size:.78rem;padding:5px 11px;border-radius:999px;background:#fff;border:1px solid var(--line);color:var(--muted);}
.pill b{color:var(--ink);font-weight:600;}

/* ---- suggestions */
.suggest-title{font-family:'Fraunces',serif;font-size:1.15rem;margin:14px 0 8px;}
div[data-testid="stButton"]>button{border-radius:14px;border:1px solid var(--line);background:#fff;color:var(--ink);
      text-align:left;padding:.7rem .9rem;font-size:.9rem;line-height:1.3;transition:all .15s;box-shadow:0 1px 0 rgba(31,27,46,.03);}
div[data-testid="stButton"]>button:hover{border-color:var(--violet);color:var(--violet);transform:translateY(-1px);
      box-shadow:0 8px 18px -10px rgba(91,61,245,.5);}

/* ---- messages */
[data-testid="stChatMessage"]{background:transparent;padding:.4rem 0;}
[data-testid="stChatMessageAvatarUser"]{background:var(--ink)!important;}
[data-testid="stChatMessageAvatarAssistant"]{background:linear-gradient(135deg,#5B3DF5,#8A5CFF)!important;}
.user-bubble{background:var(--ink);color:#fff;border-radius:18px 18px 4px 18px;padding:.7rem 1rem;display:inline-block;max-width:92%;}
.block-container div[data-testid="stVerticalBlockBorderWrapper"]{background:#fff;border:1px solid var(--line)!important;
      border-radius:18px 18px 18px 4px!important;box-shadow:0 14px 30px -22px rgba(31,27,46,.45);}
.block-container div[data-testid="stVerticalBlockBorderWrapper"] p,.block-container div[data-testid="stVerticalBlockBorderWrapper"] li{font-size:.95rem;line-height:1.55;}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important;}
div[data-testid="stButton"]>button p{white-space:normal!important;overflow:visible!important;text-overflow:clip!important;}

/* ---- jauge de confiance */
.meter{margin:10px 0 4px;}
.track{position:relative;height:9px;border-radius:99px;background:linear-gradient(90deg,#FBE3DB 0%,#FBE3DB 30%,#FCEFD2 30%,#FCEFD2 62%,#D7F0EA 62%);overflow:visible;}
.fill{height:9px;border-radius:99px;}
.tick{position:absolute;top:-4px;width:2px;height:17px;background:var(--ink);opacity:.35;border-radius:2px;}
.meta{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:9px;font-size:.78rem;color:var(--muted);}
.chip{padding:3px 10px;border-radius:99px;font-weight:600;font-size:.76rem;}
.c-confident{background:#D7F0EA;color:#13705F;} .c-uncertain{background:#FCEFD2;color:#8A5A00;}
.c-no_answer{background:#FBE3DB;color:#A83A18;} .c-human_requested{background:var(--violet-soft);color:#3F27B8;}

/* ---- sources */
.src{border:1px solid var(--line);border-radius:14px;padding:.7rem .9rem;margin:.45rem 0;background:#FFFEFB;}
.src a{font-weight:600;color:var(--violet);text-decoration:none;} .src a:hover{text-decoration:underline;}
.src .n{display:inline-grid;place-items:center;width:22px;height:22px;border-radius:7px;background:var(--violet-soft);
        color:var(--violet);font-size:.75rem;font-weight:700;margin-right:8px;}
.bars{display:flex;gap:14px;margin-top:8px;flex-wrap:wrap;}
.bar{font-size:.7rem;color:var(--muted);min-width:110px;}
.bar i{display:block;height:5px;border-radius:9px;background:#EEE7DA;margin-top:3px;overflow:hidden;}
.bar i b{display:block;height:100%;border-radius:9px;}
.ticket{background:linear-gradient(135deg,#ECE8FF,#FFF0D6);border:1px dashed var(--violet);border-radius:14px;padding:.7rem 1rem;font-size:.9rem;}

/* ---- sidebar */
section[data-testid="stSidebar"]{background:var(--sand);border-right:1px solid var(--line);}
.stat{background:#fff;border:1px solid var(--line);border-radius:14px;padding:.7rem .9rem;margin-bottom:.5rem;}
.stat .k{font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);}
.stat .v{font-family:'Fraunces',serif;font-size:1.35rem;}
[data-testid="stChatInput"]{border-radius:16px;border:1px solid var(--line);box-shadow:0 12px 28px -18px rgba(31,27,46,.5);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

LOGO = """<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
<path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M18.4 5.6l-2.1 2.1M7.7 16.3l-2.1 2.1"/><circle cx="12" cy="12" r="3.2" fill="white" stroke="none"/></svg>"""

LABEL = {"confident": "Réponse fiable", "uncertain": "À vérifier", "no_answer": "Pas de réponse fiable",
         "human_requested": "Expert humain demandé"}
COLOR = {"confident": "#1F9D8A", "uncertain": "#F5A623", "no_answer": "#E4572E", "human_requested": "#5B3DF5"}
SUGGESTIONS = [
    ("📦", "How do I load a model in 4-bit using bitsandbytes?"),
    ("🧠", "What is the difference between WordPiece and Unigram tokenization?"),
    ("🛠️", "My training crashed after 6 hours. How do I resume from the last checkpoint?"),
    ("🍎", "I'm on a MacBook with an M2 chip. How do I avoid unsupported-operator crashes?"),
    ("💬", "How do I format a conversation for a chat model?"),
    ("🧩", "How can I move optimizer states to the CPU with DeepSpeed?"),
]


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


# ----------------------------------------------------------------------------- barre latérale
with st.sidebar:
    st.markdown("### ✨ Lumen")
    st.caption("Assistant de support · Hugging Face Transformers")
    with st.expander("⚙️ Réglages", expanded=False):
        profiles = ["hf", "offline"]
        profile = st.selectbox("Profil", profiles, index=profiles.index(os.environ.get("SUPPORTBOT_PROFILE", "hf")),
                               help="hf : E5 + cross-encoder. offline : sans téléchargement.")
        llms = ["extractive", "openai", "hf-local"]
        llm_backend = st.selectbox("Générateur", llms, index=llms.index(os.environ.get("SUPPORTBOT_LLM", "extractive")),
                                   help="extractive : sans LLM. openai : Ollama / OpenAI / Groq. hf-local : modèle local.")
        cfg0 = load_config(profile)
        llm_model = st.text_input("Modèle LLM", cfg0["llm"]["model"] if llm_backend == "openai" else cfg0["llm"]["hf_model"],
                                  disabled=llm_backend == "extractive")
        base_url = st.text_input("URL de l'API", cfg0["llm"]["base_url"], disabled=llm_backend != "openai")
    try:
        bot = get_bot(profile, llm_backend, llm_model, base_url)
    except Exception as e:  # index absent, modèle non téléchargé…
        st.error(f"Impossible de démarrer le bot : {e}")
        st.stop()
    c = bot.cfg["confidence"]
    esc = resolve(bot.cfg, "log_dir") / bot.cfg["logging"]["escalations_file"]
    tickets = [json.loads(l) for l in esc.read_text(encoding="utf-8").splitlines()] if esc.exists() else []
    st.markdown(f'<div class="stat"><div class="k">Passages indexés</div><div class="v">{len(bot.index.chunks)}</div></div>'
                f'<div class="stat"><div class="k">Seuils de confiance</div><div class="v">{c["tau_low"]:.2f} · {c["tau_high"]:.2f}</div></div>'
                f'<div class="stat"><div class="k">Tickets d\'escalade</div><div class="v">{len(tickets)}</div></div>',
                unsafe_allow_html=True)
    for t in tickets[-4:][::-1]:
        st.caption(f"🎫 **{t['ticket_id']}** · {t['question'][:46]}")
    if st.button("↺  Nouvelle conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------------------------------------------------------------------- en-tête
st.markdown(f"""
<div class="hero"><div class="logo">{LOGO}</div>
<div><h1>Lumen</h1><div class="tag">Posez une question technique : la réponse vient de la documentation, avec ses sources et un niveau de confiance.</div></div></div>
<div class="pills"><span class="pill"><b>RAG</b> hybride dense + BM25</span><span class="pill"><b>Re-ranking</b> des passages</span>
<span class="pill"><b>Confiance</b> calibrée</span><span class="pill"><b>Escalade</b> vers un humain</span></div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []


def history_pairs() -> list[tuple[str, str]]:
    msgs, pairs = st.session_state.messages, []
    for i in range(0, len(msgs) - 1, 2):
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
            pairs.append((msgs[i]["content"], msgs[i + 1]["content"]))
    return pairs


def meter_html(m: dict) -> str:
    conf, dec = m["confidence"], m["decision"]
    lo, hi = bot.cfg["confidence"]["tau_low"], bot.cfg["confidence"]["tau_high"]
    return (f'<div class="meter"><div class="track"><div class="fill" style="width:{conf * 100:.0f}%;background:{COLOR[dec]}"></div>'
            f'<span class="tick" style="left:{lo * 100:.0f}%"></span><span class="tick" style="left:{hi * 100:.0f}%"></span></div>'
            f'<div class="meta"><span class="chip c-{dec}">{LABEL[dec]}</span><span>confiance <b>{conf:.0%}</b></span>'
            f'<span>· {m["latency"].get("total", 0):.0f} ms</span></div></div>')


def source_html(i: int, s: dict) -> str:
    def bar(label, val, color):
        return f'<div class="bar">{label} <b>{val:.2f}</b><i><b style="width:{max(0, min(val, 1)) * 100:.0f}%;background:{color}"></b></i></div>'
    return (f'<div class="src"><span class="n">{i}</span><a href="{html.escape(s["url"])}" target="_blank">{html.escape(s["title"])}</a>'
            f'<div class="bars">{bar("similarité", s["dense"], "#5B3DF5")}{bar("BM25", min(s["bm25"] / 15, 1), "#F5A623")}'
            f'{bar("re-ranking", s["rerank"], "#1F9D8A")}</div></div>')


def display_text(text: str) -> str:
    """Texte affiché : on retire la liste de sources (remplacée par les cartes) et l'avertissement de
    confiance (remplacé par la jauge) ; les remarques de complexité sont conservées."""
    if "**Sources**" not in text:
        return text
    head, rest = text.split("**Sources**", 1)
    extra = [p for p in rest.split("\n\n")[1:] if p.strip() and not p.startswith("⚠️")]
    return head.strip() + ("\n\n" + "\n\n".join(extra) if extra else "")


def render_bot(m: dict, idx: int) -> None:
    with st.container(border=True):
        st.markdown(display_text(m["content"]))
        st.markdown(meter_html(m), unsafe_allow_html=True)
    if m["sources"]:
        with st.expander(f"📚 {len(m['sources'])} sources utilisées"):
            for i, s in enumerate(m["sources"], 1):
                st.markdown(source_html(i, s), unsafe_allow_html=True)
                with st.popover(f"Voir l'extrait {i}"):
                    st.text(s["text"][:900] + ("…" if len(s["text"]) > 900 else ""))
    cols = st.columns([1, 1, 3, 6])
    if cols[0].button("👍", key=f"up{idx}"):
        append_jsonl(resolve(bot.cfg, "log_dir") / "feedback.jsonl", {"ts": datetime.now(timezone.utc).isoformat(), "q": m["question"], "vote": 1})
        st.toast("Merci pour votre retour !")
    if cols[1].button("👎", key=f"down{idx}"):
        append_jsonl(resolve(bot.cfg, "log_dir") / "feedback.jsonl", {"ts": datetime.now(timezone.utc).isoformat(), "q": m["question"], "vote": -1})
        st.toast("Retour enregistré.")
    if m["escalation_suggested"]:
        if m.get("ticket"):
            cols[3].markdown(f'<div class="ticket">🎫 Ticket <b>{m["ticket"]}</b> créé : un expert vous répondra.</div>', unsafe_allow_html=True)
        elif cols[2].button("🙋 Parler à un expert", key=f"esc{idx}", help=f"Raison : {m['escalation_reason']}"):
            hist = history_pairs()[:-1]
            m["ticket"] = bot.escalate(m["question"], hist, m["escalation_reason"] or "demande de l'utilisateur")["ticket_id"]
            st.rerun()


# ----------------------------------------------------------------------------- écran d'accueil
if not st.session_state.messages:
    st.markdown('<div class="suggest-title">Par où commencer ?</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (icon, q) in enumerate(SUGGESTIONS):
        if cols[i % 3].button(f"{icon}  {q}", key=f"sg{i}", use_container_width=True):
            st.session_state.pending = q
            st.rerun()

for i, m in enumerate(st.session_state.messages):
    if m["role"] == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(f'<div class="user-bubble">{html.escape(m["content"])}</div>', unsafe_allow_html=True)
    else:
        with st.chat_message("assistant", avatar="✨"):
            render_bot(m, i)

prompt = st.chat_input("Posez votre question (ex. « How do I enable gradient checkpointing? »)")
prompt = prompt or st.session_state.pop("pending", None)
if prompt:
    hist = history_pairs()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Lumen cherche dans la documentation…"):
        r = bot.answer(prompt, hist)
    meta = {"role": "assistant", "content": r.answer, "decision": r.decision, "confidence": r.confidence,
            "latency": r.latency_ms, "question": prompt, "escalation_suggested": r.escalation_suggested,
            "escalation_reason": r.escalation_reason, "ticket": None,
            "sources": [{"title": s.chunk.doc_title + (f" › {s.chunk.section}" if s.chunk.section else ""),
                         "url": s.chunk.url, "dense": s.dense_score, "bm25": s.bm25_score,
                         "rerank": s.rerank_score or 0.0, "text": s.chunk.text} for s in r.sources]}
    if r.decision == "human_requested":                       # demande explicite : ticket immédiat
        meta["ticket"] = bot.escalate(prompt, hist, r.escalation_reason, r)["ticket_id"]
    st.session_state.messages.append(meta)
    st.rerun()
