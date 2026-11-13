"""Interface en ligne de commande :  python -m supportbot <commande> [--profile hf|offline] [--set a.b=val]

  build      télécharge (option) puis indexe le corpus
  ask        pose une question et affiche la réponse
  chat       conversation interactive (historique, escalade)
  calibrate  entraîne le re-ranker (profil offline) et calibre les seuils de confiance sur le jeu de dev
  eval       évalue sur le jeu de test et écrit results/<profil>/
  logs       statistiques sur les journaux de requêtes
"""
from __future__ import annotations

import argparse
import json
import sys

import yaml

from .config import ROOT, load_config, resolve


def _overrides(pairs: list[str]) -> dict:
    out: dict = {}
    for p in pairs or []:
        key, _, val = p.partition("=")
        d = out
        *parents, last = key.split(".")
        for k in parents:
            d = d.setdefault(k, {})
        d[last] = yaml.safe_load(val)
    return out


def cmd_build(args) -> None:
    from .chunking import chunk_corpus
    from .embeddings import make_embedder
    from .index import HybridIndex
    from .ingest import load_documents

    cfg = load_config(args.profile, _overrides(args.set))
    raw = resolve(cfg, "raw_dir")
    if not list(raw.rglob("*.md")):
        sys.exit("Corpus absent : lancez d'abord  python scripts/download_corpus.py")
    docs = load_documents(raw)
    c = cfg["chunking"]
    chunks = chunk_corpus(docs, c["max_words"], c["overlap_words"], c["min_words"])
    print(f"{len(docs)} documents → {len(chunks)} passages ; embeddings « {cfg['embedding']['model']} » …")
    index = HybridIndex.build(chunks, make_embedder(cfg))
    index.save(resolve(cfg, "index_dir"), cfg)
    proc = resolve(cfg, "processed_dir")
    proc.mkdir(parents=True, exist_ok=True)
    with open(proc / "chunks.jsonl", "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch.to_dict(), ensure_ascii=False) + "\n")
    print(f"Index écrit dans {resolve(cfg, 'index_dir')}")


def _bot(args):
    from .pipeline import SupportBot
    return SupportBot.from_config(args.profile, _overrides(args.set))


def _print_response(r) -> None:
    badge = {"confident": "🟢", "uncertain": "🟠", "no_answer": "🔴", "human_requested": "🙋"}[r.decision]
    print(f"\n{r.answer}\n\n{badge} {r.decision} (confiance {r.confidence:.0%}) — "
          f"{r.latency_ms.get('total', 0):.0f} ms")
    if r.escalation_suggested and r.decision != "human_requested":
        print(f"   ↳ escalade suggérée : {r.escalation_reason} (tapez /escalade pour créer un ticket)")


def cmd_ask(args) -> None:
    bot = _bot(args)
    r = bot.answer(" ".join(args.question), log=not args.no_log)
    _print_response(r)


def cmd_chat(args) -> None:
    bot = _bot(args)
    history: list[tuple[str, str]] = []
    last = None
    print("Assistant support Hugging Face Transformers — /quit pour sortir, /escalade pour parler à un humain.\n")
    while True:
        try:
            q = input("vous › ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in {"/quit", "/q"}:
            break
        if q.startswith("/escalad") and last:
            t = bot.escalate(last.question, history, "demande de l'utilisateur", last)
            print(f"Ticket {t['ticket_id']} créé (logs/{bot.cfg['logging']['escalations_file']}).")
            continue
        last = bot.answer(q, history)
        _print_response(last)
        if last.decision == "human_requested":
            t = bot.escalate(q, history, last.escalation_reason, last)
            print(f"Ticket {t['ticket_id']} créé.")
        history.append((q, last.answer))


def cmd_calibrate(args) -> None:
    from .evaluation import calibrate
    calibrate(args.profile, _overrides(args.set))


def cmd_eval(args) -> None:
    from .evaluation import evaluate
    evaluate(args.profile, _overrides(args.set), split=args.split)


def cmd_logs(args) -> None:
    from .logstats import summarize
    cfg = load_config(args.profile)
    summarize(resolve(cfg, "log_dir") / cfg["logging"]["queries_file"])


def main() -> None:
    ap = argparse.ArgumentParser(prog="supportbot", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, help_ in [("build", cmd_build, "indexer le corpus"), ("ask", cmd_ask, "poser une question"),
                            ("chat", cmd_chat, "conversation interactive"),
                            ("calibrate", cmd_calibrate, "entraîner/calibrer sur le dev"),
                            ("eval", cmd_eval, "évaluer"), ("logs", cmd_logs, "statistiques des logs")]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--profile", default="hf", help="hf | offline | chemin d'un YAML")
        p.add_argument("--set", action="append", help="surcharge de config, ex. --set llm.backend=openai")
        p.set_defaults(fn=fn)
        if name == "ask":
            p.add_argument("question", nargs="+")
            p.add_argument("--no-log", action="store_true")
        if name == "eval":
            p.add_argument("--split", default="test", choices=["test", "dev", "all"])
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
