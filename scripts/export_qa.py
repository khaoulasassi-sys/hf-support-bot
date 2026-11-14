#!/usr/bin/env python3
"""Exporte toutes les questions du jeu d'évaluation avec la réponse du chatbot -> results/<profil>/questions_reponses.md"""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from supportbot.pipeline import SupportBot
from supportbot.evaluation import load_questions

profile = sys.argv[1] if len(sys.argv) > 1 else "hf"
bot = SupportBot.from_config(profile)
LABEL = {"in_scope": "Question couverte par la documentation", "out_of_scope": "Hors périmètre (le bot doit refuser)",
         "ambiguous": "Ambiguë / complexe (le bot doit escalader)", "human": "Demande d'un humain"}
out = [f"# Questions et réponses du chatbot (profil {profile})\n",
       "Réponses générées par le chatbot en mode extractif (sans LLM). Page cible = page de la documentation Hugging Face qui contient la réponse.\n"]
for cat in ("in_scope", "out_of_scope", "ambiguous", "human"):
    qs = [q for q in load_questions() if q["category"] == cat]
    out.append(f"\n## {LABEL[cat]} ({len(qs)} questions)\n")
    for q in qs:
        r = bot.answer(q["question"], log=False)
        gold = f"\n**Page cible :** {', '.join(q['gold_docs'])}  \n" if cat == "in_scope" else ""
        body = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", r.answer.split("**Sources**")[0].strip())
        srcs = "\n".join(f"- {s.chunk.doc_title} > {s.chunk.section} — {s.chunk.url}" for s in r.sources)
        out.append(f"### {q['id']} — {q['question'].splitlines()[0][:150]}\n{gold}\n**Réponse du bot** (décision : {r.decision}, confiance {r.confidence:.0%}) :\n\n{body}\n\n" + (f"**Sources :**\n{srcs}\n" if srcs else ""))
dest = ROOT / "results" / bot.cfg["profile"] / "questions_reponses.md"
dest.write_text("\n".join(out), encoding="utf-8")
print(dest, len(out))
