#!/usr/bin/env python3
"""Rejoue des conversations scénarisées et écrit results/<profil>/conversations.json (pour le rapport)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportbot.pipeline import SupportBot

profile = sys.argv[1] if len(sys.argv) > 1 else "hf"
llm = sys.argv[2] if len(sys.argv) > 2 else "extractive"
bot = SupportBot.from_config(profile, {"llm": {"backend": llm}})
SCENARIOS = {
 "Question simple (réponse confiante)": ["How do I load a model in 4-bit using bitsandbytes?"],
 "Question de suivi (historique)": ["How do I enable gradient accumulation in the Trainer?", "and how does it affect the loss?"],
 "Hors périmètre (refus)": ["How do I deploy my web app on AWS Lambda?"],
 "Question vague (incertitude / escalade)": ["My model is slow."],
 "Demande d'un humain": ["Can I speak to a human please?"],
}
out = []
for title, qs in SCENARIOS.items():
    hist, turns = [], []
    for q in qs:
        r = bot.answer(q, hist, log=False)
        turns.append({"user": q, "bot": r.answer, "decision": r.decision, "confidence": round(r.confidence, 3),
                      "escalation": r.escalation_suggested, "retrieval_query": r.retrieval_query})
        hist.append((q, r.answer))
    out.append({"scenario": title, "turns": turns})
d = Path(__file__).resolve().parents[1] / "results" / bot.cfg["profile"]
(d / f"conversations_{llm}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print("ok", d)
