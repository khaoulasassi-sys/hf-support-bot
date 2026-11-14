"""Calibration (jeu de DEV) et évaluation (jeu de TEST) du chatbot.

calibrate : 1. (profil « feature ») entraîne le re-ranker sur les candidats des questions de dev ;
            2. calibre tau_high / tau_low avec des scores *hors-échantillon* (validation croisée)
               pour ne pas être optimiste : on vise une précision cible des réponses « confiantes »
               et un taux maximal de refus à tort.
evaluate  : mesure sur le jeu de test — retrieval (ablation BM25 / dense / hybride / hybride+rerank),
            qualité des réponses, gestion de l'incertitude, escalade, latence — et écrit
            results/<profil>/{metrics.json, per_question_<split>.jsonl, tables.csv, figures/}.
"""
from __future__ import annotations

import json
import statistics as st
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import ROOT, load_config, resolve
from .embeddings import make_embedder
from .index import HybridIndex
from .pipeline import SupportBot
from .reranker import FeatureReranker, make_reranker, train_feature_reranker, FEATURE_NAMES
from .types import Candidate, Chunk

QUESTIONS = ROOT / "data" / "eval" / "questions.jsonl"
TARGET_PRECISION = 0.90          # précision visée pour les réponses « confiantes »
MAX_WRONGFUL_ABSTENTION = 0.15   # part maximale de questions couvertes refusées à tort


def load_questions(split: str = "all") -> list[dict]:
    qs = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    return qs if split == "all" else [q for q in qs if q["split"] == split]


def is_relevant(chunk: Chunk, q: dict) -> bool:
    """Un passage est pertinent s'il appartient à une page cible ET contient un des mots-clés attendus."""
    if q["category"] != "in_scope" or chunk.doc_id not in q["gold_docs"]:
        return False
    low = chunk.text.lower()
    return any(k.lower() in low for k in q["keywords"])


def _index(cfg: dict) -> HybridIndex:
    return HybridIndex.load(resolve(cfg, "index_dir"), make_embedder(cfg))


def _pctl(xs: list[float], p: float) -> float:
    return float(np.percentile(xs, p)) if xs else float("nan")


# ====================================================================================== calibration
def calibrate(profile: str, overrides: dict | None = None) -> dict:
    cfg = load_config(profile, overrides)
    out_dir = ROOT / "results" / cfg["profile"]
    out_dir.mkdir(parents=True, exist_ok=True)
    index = _index(cfg)
    r = cfg["retrieval"]
    dev = [q for q in load_questions("dev") if q["category"] in ("in_scope", "out_of_scope", "ambiguous")]
    cands = {q["id"]: index.search(q["question"], r["mode"], r["k_dense"], r["k_bm25"], r["k_candidates"], r["rrf_k"])
             for q in dev}
    top_n = cfg["rerank"]["top_n"]
    info: dict = {"n_dev": len(dev)}

    def score_with(reranker, q) -> tuple[float, bool]:
        top = reranker.rerank(q["question"], [Candidate(c.chunk, c.dense_score, c.bm25_score, c.fused_score)
                                              for c in cands[q["id"]]], top_n)
        conf = top[0].rerank_score if top else 0.0
        ok = q["category"] == "in_scope" and any(is_relevant(c.chunk, q) for c in top)
        return float(conf), ok

    items: list[tuple[float, bool, str]] = []
    if cfg["rerank"]["backend"] == "feature":
        K = 5
        folds = [dev[i::K] for i in range(K)]
        for k in range(K):                                  # scores hors-échantillon
            train = [q for j, f in enumerate(folds) if j != k for q in f]
            model_path = out_dir / f"_fold{k}.joblib"
            train_feature_reranker([(q["question"], cands[q["id"]], [int(is_relevant(c.chunk, q)) for c in cands[q["id"]]])
                                    for q in train], model_path)
            rr = FeatureReranker(model_path)
            for q in folds[k]:
                conf, ok = score_with(rr, q)
                items.append((conf, ok, q["category"]))
            model_path.unlink()
        model, tinfo = train_feature_reranker(
            [(q["question"], cands[q["id"]], [int(is_relevant(c.chunk, q)) for c in cands[q["id"]]]) for q in dev],
            ROOT / cfg["rerank"]["model"])
        info["feature_reranker"] = tinfo
    else:
        rr = make_reranker(cfg)
        items = [(*score_with(rr, q), q["category"]) for q in dev]

    confs = np.array([c for c, _, _ in items])
    oks = np.array([o for _, o, _ in items])
    # tau_high : plus petit seuil dont les réponses « confiantes » atteignent la précision visée
    tau_high = None
    for t in sorted(set(confs.tolist())):
        sel = confs >= t
        if sel.sum() >= 3 and oks[sel].mean() >= TARGET_PRECISION:
            tau_high = t
            break
    if tau_high is None:
        tau_high = float(confs.max())
    # tau_low : plus grand seuil qui ne refuse à tort qu'au plus MAX_WRONGFUL_ABSTENTION des questions couvertes
    good = np.sort(confs[oks])
    tau_low = float(good[int(np.floor(MAX_WRONGFUL_ABSTENTION * len(good)))]) if len(good) else 0.0
    tau_low = min(tau_low, tau_high)
    info.update({"confidence": {"tau_high": round(float(tau_high), 4), "tau_low": round(float(tau_low), 4)},
                 "target_precision": TARGET_PRECISION, "max_wrongful_abstention": MAX_WRONGFUL_ABSTENTION,
                 "dev_confident_precision": float(oks[confs >= tau_high].mean()) if (confs >= tau_high).any() else None,
                 "dev_confident_coverage": float((confs >= tau_high).mean())})
    (out_dir / "calibration.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    return info


# ====================================================================================== évaluation
def _rank_metrics(ranked: list[Candidate], q: dict) -> dict:
    rel = [is_relevant(c.chunk, q) for c in ranked]
    doc = [c.chunk.doc_id in q["gold_docs"] for c in ranked]
    first = next((i + 1 for i, r in enumerate(rel) if r), None)
    return {"hit@1": any(rel[:1]), "hit@3": any(rel[:3]), "hit@5": any(rel[:5]),
            "mrr": 1.0 / first if first else 0.0,
            "doc_hit@3": any(doc[:3]),
            "offtopic@3": (3 - sum(doc[:3])) / 3 if len(doc) >= 3 else (len(doc) - sum(doc[:3])) / max(len(doc), 1)}


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


def evaluate(profile: str, overrides: dict | None = None, split: str = "test") -> dict:
    cfg = load_config(profile, overrides)
    out_dir = ROOT / "results" / cfg["profile"]
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    bot = SupportBot.from_config(cfg)
    r = cfg["retrieval"]
    qs = load_questions(split)
    metrics: dict = {"profile": cfg["profile"], "split": split, "n_questions": len(qs),
                     "config": {k: cfg[k] for k in ("embedding", "retrieval", "rerank", "confidence", "chunking")},
                     "index": json.loads((resolve(cfg, "index_dir") / "meta.json").read_text(encoding="utf-8"))}

    # ---- 1. retrieval : ablation (in_scope, anglais) ----------------------------------------
    ins = [q for q in qs if q["category"] == "in_scope" and q["lang"] == "en"]
    variants: dict[str, list[dict]] = defaultdict(list)
    cand_recall = []
    for q in ins:
        for name, mode in (("BM25 seul", "bm25"), ("Dense seul", "dense"), ("Hybride (RRF)", "hybrid")):
            c = bot.index.search(q["question"], mode, r["k_dense"], r["k_bm25"], r["k_candidates"], r["rrf_k"])
            variants[name].append(_rank_metrics(c, q))
        c = bot.index.search(q["question"], r["mode"], r["k_dense"], r["k_bm25"], r["k_candidates"], r["rrf_k"])
        cand_recall.append(any(is_relevant(x.chunk, q) for x in c))
        reranked = bot.reranker.rerank(q["question"], c, len(c))
        variants["Hybride + re-ranking"].append(_rank_metrics(reranked, q))
    metrics["retrieval"] = {name: {k: round(_mean(rows, k), 4) for k in rows[0]} for name, rows in variants.items()}
    metrics["retrieval"]["_candidate_recall@%d" % r["k_candidates"]] = round(float(np.mean(cand_recall)), 4)
    metrics["n_in_scope_en"] = len(ins)

    # ---- 2. pipeline complet ----------------------------------------------------------------
    per_q = []
    for q in qs:
        t0 = time.perf_counter()
        resp = bot.answer(q["question"], log=False)
        cands = bot.retrieve(resp.retrieval_query)
        rel_in_cands = any(is_relevant(c.chunk, q) for c in cands)
        rel_in_top = any(is_relevant(c.chunk, q) for c in resp.sources) if resp.sources else False
        if q["category"] == "in_scope":
            top_n = bot.reranker.rerank(resp.retrieval_query, cands, cfg["rerank"]["top_n"])
            rel_in_top_any = any(is_relevant(c.chunk, q) for c in top_n)
            if not rel_in_cands:
                failure = "retrieval_miss"
            elif not rel_in_top_any:
                failure = "rerank_miss"
            elif resp.decision == "no_answer":
                failure = "wrongful_abstention"
            else:
                failure = "ok"
            correct = resp.decision in ("confident", "uncertain") and rel_in_top
        else:
            rel_in_top_any = False
            exp = "human_requested" if q["category"] == "human" else None
            if exp:
                failure = "ok" if resp.decision == exp else "human_not_detected"
            else:
                failure = "confident_hallucination_risk" if resp.decision == "confident" else "ok"
            correct = False
        per_q.append({"id": q["id"], "category": q["category"], "lang": q["lang"], "question": q["question"][:160],
                      "decision": resp.decision, "confidence": round(resp.confidence, 4),
                      "escalation_suggested": resp.escalation_suggested, "answered": resp.decision in ("confident", "uncertain"),
                      "correct": correct, "failure": failure, "top1_doc": resp.sources[0].chunk.doc_id if resp.sources else None,
                      "gold_docs": q.get("gold_docs"), "sources": [s.chunk.chunk_id for s in resp.sources],
                      "latency_ms": resp.latency_ms})
    with open(out_dir / f"per_question_{split}.jsonl", "w", encoding="utf-8") as f:
        for row in per_q:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def cat(c, lang=None):
        return [p for p in per_q if p["category"] == c and (lang is None or p["lang"] == lang)]

    ins_all = cat("in_scope", "en")
    answered = [p for p in ins_all if p["answered"]]
    conf_ans = [p for p in ins_all if p["decision"] == "confident"]
    m: dict = {}
    m["in_scope"] = {
        "n": len(ins_all),
        "coverage": round(len(answered) / len(ins_all), 4),
        "answer_precision": round(sum(p["correct"] for p in answered) / max(len(answered), 1), 4),
        "overall_correct": round(sum(p["correct"] for p in ins_all) / len(ins_all), 4),
        "confident_rate": round(len(conf_ans) / len(ins_all), 4),
        "confident_precision": round(sum(p["correct"] for p in conf_ans) / max(len(conf_ans), 1), 4),
        "uncertain_rate": round(sum(p["decision"] == "uncertain" for p in ins_all) / len(ins_all), 4),
        "wrongful_abstention_rate": round(sum(p["decision"] == "no_answer" for p in ins_all) / len(ins_all), 4),
        "escalation_rate": round(sum(p["escalation_suggested"] for p in ins_all) / len(ins_all), 4),
    }
    oos, amb, hum = cat("out_of_scope"), cat("ambiguous"), cat("human")
    m["out_of_scope"] = {"n": len(oos),
                         "abstention_rate": round(sum(p["decision"] == "no_answer" for p in oos) / len(oos), 4),
                         "not_confident_rate": round(sum(p["decision"] != "confident" for p in oos) / len(oos), 4),
                         "escalation_rate": round(sum(p["escalation_suggested"] for p in oos) / len(oos), 4)}
    m["ambiguous"] = {"n": len(amb),
                      "escalation_rate": round(sum(p["escalation_suggested"] for p in amb) / len(amb), 4),
                      "not_confident_rate": round(sum(p["decision"] != "confident" for p in amb) / len(amb), 4)}
    m["human_request"] = {"n": len(hum), "detected_rate": round(sum(p["decision"] == "human_requested" for p in hum) / max(len(hum), 1), 4)}
    risky = oos + amb
    m["safety"] = {"n": len(risky),
                   "confident_hallucination_risk": round(sum(p["decision"] == "confident" for p in risky) / len(risky), 4),
                   "escalation_recall": round(sum(p["escalation_suggested"] for p in risky) / len(risky), 4),
                   "escalation_precision": round(
                       sum(p["escalation_suggested"] for p in risky) /
                       max(sum(p["escalation_suggested"] for p in risky + ins_all), 1), 4)}
    fr = cat("in_scope", "fr")
    m["in_scope_fr"] = {"n": len(fr),
                        "coverage": round(sum(p["answered"] for p in fr) / max(len(fr), 1), 4),
                        "overall_correct": round(sum(p["correct"] for p in fr) / max(len(fr), 1), 4)} if fr else {"n": 0}
    allq = [p for p in per_q if p["category"] != "human"]
    m["global"] = {"n": len(per_q), "uncertain_rate": round(sum(p["decision"] == "uncertain" for p in allq) / len(allq), 4),
                   "no_answer_rate": round(sum(p["decision"] == "no_answer" for p in allq) / len(allq), 4),
                   "confident_rate": round(sum(p["decision"] == "confident" for p in allq) / len(allq), 4),
                   "escalation_rate": round(sum(p["escalation_suggested"] for p in allq) / len(allq), 4)}
    lat = defaultdict(list)
    for p in per_q:
        for k, v in p["latency_ms"].items():
            lat[k].append(v)
    m["latency_ms"] = {k: {"mean": round(st.mean(v), 1), "p50": round(_pctl(v, 50), 1), "p95": round(_pctl(v, 95), 1)}
                       for k, v in lat.items()}
    # fiabilité : précision par tranche de confiance (in_scope anglais + hors périmètre + ambigu)
    bins = [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]
    rel_rows = []
    for lo, hi in bins:
        sel = [p for p in per_q if p["category"] in ("in_scope", "out_of_scope", "ambiguous") and p["lang"] == "en"
               and lo <= p["confidence"] < hi]
        rel_rows.append({"bin": f"{lo:.1f}-{min(hi, 1):.1f}", "n": len(sel),
                         "precision": round(sum(p["correct"] for p in sel) / len(sel), 4) if sel else None})
    m["reliability"] = rel_rows
    fails = defaultdict(int)
    for p in per_q:
        fails[p["failure"]] += 1
    m["failure_breakdown"] = dict(fails)
    metrics.update(m)
    (out_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_tables(metrics, out_dir, split)
    _write_figures(metrics, per_q, out_dir, split)
    print(json.dumps({k: metrics[k] for k in ("retrieval", "in_scope", "out_of_scope", "ambiguous", "human_request",
                                               "safety", "in_scope_fr", "global", "failure_breakdown")}, indent=2, ensure_ascii=False))
    return metrics


# ====================================================================================== tableaux & figures
def _write_tables(m: dict, out_dir: Path, split: str) -> None:
    import csv
    with open(out_dir / f"table_retrieval_{split}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        keys = ["hit@1", "hit@3", "hit@5", "mrr", "offtopic@3"]
        w.writerow(["variant"] + keys)
        for name, row in m["retrieval"].items():
            if not name.startswith("_"):
                w.writerow([name] + [row[k] for k in keys])
    with open(out_dir / f"table_pipeline_{split}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["groupe", "metrique", "valeur"])
        for g in ("in_scope", "out_of_scope", "ambiguous", "human_request", "safety", "in_scope_fr", "global"):
            for k, v in m[g].items():
                w.writerow([g, k, v])


def _write_figures(m: dict, per_q: list[dict], out_dir: Path, split: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    names = [n for n in m["retrieval"] if not n.startswith("_")]
    # (a) ablation retrieval
    fig, ax = plt.subplots(figsize=(7, 3.4))
    w = 0.2
    for i, k in enumerate(["hit@1", "hit@3", "hit@5", "mrr"]):
        ax.bar(np.arange(len(names)) + i * w, [m["retrieval"][n][k] for n in names], w,
               label={"hit@1": "Hit@1", "hit@3": "Hit@3", "hit@5": "Hit@5", "mrr": "MRR"}[k])
    ax.set_xticks(np.arange(len(names)) + 1.5 * w)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend(ncol=4, fontsize=8, loc="upper left", frameon=False)
    ax.set_title("Retrieval : apport de l'hybride et du re-ranking (jeu de test)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "fig_retrieval.png", dpi=200)
    plt.close(fig)
    # (b) distribution des scores de confiance par catégorie
    fig, ax = plt.subplots(figsize=(7, 3.2))
    groups = [("Questions couvertes", [p["confidence"] for p in per_q if p["category"] == "in_scope" and p["lang"] == "en"], "#2a9d8f"),
              ("Hors périmètre", [p["confidence"] for p in per_q if p["category"] == "out_of_scope"], "#e76f51"),
              ("Ambiguës / complexes", [p["confidence"] for p in per_q if p["category"] == "ambiguous"], "#e9c46a")]
    bins = np.linspace(0, 1, 11)
    for i, (lab, vals, col) in enumerate(groups):
        h, _ = np.histogram(vals, bins=bins)
        ax.bar(bins[:-1] + 0.033 * i + 0.005, h / max(len(vals), 1), 0.033, label=f"{lab} (n={len(vals)})", color=col)
    th, tl = m["config"]["confidence"]["tau_high"], m["config"]["confidence"]["tau_low"]
    ax.axvline(th, color="k", ls="--", lw=1)
    ax.axvline(tl, color="k", ls=":", lw=1)
    ax.text(th, ax.get_ylim()[1] * 0.93, " τ_high", fontsize=8)
    ax.text(tl, ax.get_ylim()[1] * 0.93, " τ_low", fontsize=8)
    ax.set_xlabel("Score de confiance (probabilité de pertinence du meilleur passage)")
    ax.set_ylabel("Part des questions")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "fig_confidence.png", dpi=200)
    plt.close(fig)
    # (c) décisions par catégorie
    fig, ax = plt.subplots(figsize=(7, 3.0))
    cats = [("in_scope", "Couvertes (EN)"), ("out_of_scope", "Hors périmètre"), ("ambiguous", "Ambiguës")]
    cols = {"confident": "#2a9d8f", "uncertain": "#e9c46a", "no_answer": "#e76f51"}
    left = np.zeros(len(cats))
    for dec, lab in [("confident", "Confiant"), ("uncertain", "Incertain"), ("no_answer", "Refus (je ne sais pas)")]:
        vals = []
        for c, _ in cats:
            rows = [p for p in per_q if p["category"] == c and (c != "in_scope" or p["lang"] == "en")]
            vals.append(sum(p["decision"] == dec for p in rows) / len(rows))
        ax.barh([n for _, n in cats], vals, left=left, label=lab, color=cols[dec])
        left += np.array(vals)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.38))
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "fig_decisions.png", dpi=200)
    plt.close(fig)
