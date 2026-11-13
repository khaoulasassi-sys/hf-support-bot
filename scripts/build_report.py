#!/usr/bin/env python3
"""Génère report/results_tables.tex à partir de results/<profil>/ puis compile report/main.tex."""
import json, os, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
profile = sys.argv[1] if len(sys.argv) > 1 else "offline"
R = ROOT / "results" / profile
m = json.loads((R / "metrics_test.json").read_text())
cal = json.loads((R / "calibration.json").read_text())
pct = lambda x: f"{100 * x:.0f}\\,\\%"
f2 = lambda x: f"{x:.2f}"

def tex_escape(s):
    return s.replace("\\", "").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#").replace("$", "\\$")

rows = []
for name, r in m["retrieval"].items():
    if name.startswith("_"): continue
    b = "\\textbf{%s}" if name == "Hybride + re-ranking" else "%s"
    rows.append(f"{b % tex_escape(name)} & {f2(r['hit@1'])} & {f2(r['hit@3'])} & {f2(r['hit@5'])} & {f2(r['mrr'])} & {pct(r['offtopic@3'])} \\\\")
t1 = "\\begin{tabular}{lrrrrr}\\toprule\nVariante & Hit@1 & Hit@3 & Hit@5 & MRR & Hors-sujet@3 \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}"

i, o, a, h, s, g, fr = (m[k] for k in ("in_scope", "out_of_scope", "ambiguous", "human_request", "safety", "global", "in_scope_fr"))
t2 = f"""\\begin{{tabular}}{{llr}}\\toprule
Groupe & Indicateur & Valeur \\\\\\midrule
Questions couvertes (EN, $n={i['n']}$) & Taux de réponse (confiant + incertain) & {pct(i['coverage'])} \\\\
 & Précision des réponses données & {pct(i['answer_precision'])} \\\\
 & Précision des réponses « confiantes » & {pct(i['confident_precision'])} \\\\
 & Questions correctement traitées (sur $n$) & {pct(i['overall_correct'])} \\\\
 & Refus à tort (« je ne sais pas ») & {pct(i['wrongful_abstention_rate'])} \\\\
 & Réponses incertaines (avec réserve) & {pct(i['uncertain_rate'])} \\\\\\midrule
Hors périmètre ($n={o['n']}$) & Refus (aucune réponse inventée) & {pct(o['abstention_rate'])} \\\\
Ambiguës / complexes ($n={a['n']}$) & Escalade proposée & {pct(a['escalation_rate'])} \\\\
Demande d'un humain ($n={h['n']}$) & Détection & {pct(h['detected_rate'])} \\\\
Hors périmètre + ambiguës ($n={s['n']}$) & Réponse « confiante » à tort & {pct(s['confident_hallucination_risk'])} \\\\
 & Rappel de l'escalade & {pct(s['escalation_recall'])} \\\\\\midrule
Français ($n={fr['n']}$) & Questions correctement traitées & {pct(fr['overall_correct'])} \\\\
Toutes catégories ($n={g['n']}$) & Taux d'incertitude / refus / escalade & {pct(g['uncertain_rate'])} / {pct(g['no_answer_rate'])} / {pct(g['escalation_rate'])} \\\\
\\bottomrule\\end{{tabular}}"""
lat = m["latency_ms"]
t3 = ("\\begin{tabular}{lrrr}\\toprule\nÉtape & Moyenne (ms) & Médiane & P95 \\\\\\midrule\n" +
      "\n".join(f"{k} & {v['mean']} & {v['p50']} & {v['p95']} \\\\" for k, v in lat.items() if k in ("retrieve", "rerank", "generate", "total")) +
      "\n\\bottomrule\\end{tabular}")
rel = ("\\begin{tabular}{lrr}\\toprule\nConfiance & $n$ & Précision observée \\\\\\midrule\n" +
       "\n".join(f"{r['bin']} & {r['n']} & {'--' if r['precision'] is None else pct(r['precision'])} \\\\" for r in m["reliability"]) +
       "\n\\bottomrule\\end{tabular}")
fb = m["failure_breakdown"]
t4 = ("\\begin{tabular}{lr}\\toprule\nCatégorie de résultat & Nombre \\\\\\midrule\n" +
      "\n".join(f"{tex_escape(k)} & {v} \\\\" for k, v in sorted(fb.items(), key=lambda kv: -kv[1])) + "\n\\bottomrule\\end{tabular}")

def clean(txt):  # retire emojis / caractères hors latin-1 ; échappe pour lstlisting
    return re.sub(r"[^\x00-\xff]", "", txt).replace("\u00a0", " ")
conv = []
cj = R / "conversations_extractive.json"
for sc in json.loads(cj.read_text()):
    conv.append(f"\\paragraph{{{tex_escape(sc['scenario'])}}}")
    for t in sc["turns"]:
        bot = clean(t["bot"]); bot = bot.split("**Sources**")[0].strip()
        bot = bot[:650] + (" [...]" if len(bot) > 650 else "")
        conv.append("\\begin{lstlisting}\n" + clean(t["user"]) + "\n\\end{lstlisting}\n\\begin{lstlisting}[backgroundcolor=\\color{black!5}]\n" + bot.replace("**", "") +
                    f"\n\\end{{lstlisting}}\n\\noindent\\textit{{Décision : {tex_escape(t['decision'])}, confiance {100*t['confidence']:.0f}\\%" + (", escalade proposée" if t["escalation"] else "") + "}.\n")
per = [json.loads(l) for l in (R / "per_question_test.jsonl").read_text().splitlines()]
fail_rows = [p for p in per if p["failure"] not in ("ok",)]
for key, macro in (("REPO_URL", "repourl"), ("AUTHORS", "authors")):
    if os.environ.get(key):
        (ROOT / "report" / f"{macro}.tex").write_text(os.environ[key])
repo = (ROOT / "report" / "repourl.tex").read_text().strip()
(ROOT / "report" / "results_tables.tex").write_text("\n".join([
    f"\\newcommand{{\\RepoURL}}{{{repo}}}",
    f"\\newcommand{{\\TabRetrieval}}{{{t1}}}", f"\\newcommand{{\\TabPipeline}}{{{t2}}}", f"\\newcommand{{\\TabLatency}}{{{t3}}}",
    f"\\newcommand{{\\TabReliability}}{{{rel}}}", f"\\newcommand{{\\TabFailures}}{{{t4}}}",
    f"\\newcommand{{\\TauHigh}}{{{cal['confidence']['tau_high']:.2f}}}", f"\\newcommand{{\\TauLow}}{{{cal['confidence']['tau_low']:.2f}}}",
    f"\\newcommand{{\\NChunks}}{{{m['index']['n_chunks']}}}", f"\\newcommand{{\\NDocs}}{{{m['index']['n_docs']}}}",
    f"\\newcommand{{\\CandRecall}}{{{pct(m['retrieval']['_candidate_recall@20'])}}}",
    f"\\newcommand{{\\ProfileName}}{{{profile}}}",
    ]))
(ROOT / "report" / "conversations.tex").write_text("\n".join(conv))
env = dict(os.environ)
for key, macro in (("REPO_URL", "repourl"), ("AUTHORS", "authors")):
    if os.environ.get(key):
        (ROOT / "report" / f"{macro}.tex").write_text(os.environ[key])
for _ in range(2):
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=ROOT / "report", capture_output=True)
if not (ROOT / "report" / "main.pdf").exists():
    print(r.stdout[-3000:].decode("utf8","replace")); sys.exit(1)
print("rapport : report/main.pdf")
