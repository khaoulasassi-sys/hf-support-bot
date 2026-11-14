# Projet 3 — Chatbot de support technique (RAG + re-ranking + incertitude)

Assistant de support pour la bibliothèque **Hugging Face Transformers** : recherche hybride (embeddings « passage » + BM25),
re-ranking, génération par LLM avec sources, **score de confiance**, refus/avertissement, **escalade vers un humain**,
interface de chat, journalisation et évaluation. Rapport : `report/main.pdf`.

## Installation (5 min)
```bash
python -m venv .venv && source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_corpus.py                        # 68 pages de doc (déjà fournies dans data/raw)
```
## Utilisation
| Commande | Rôle |
|---|---|
| `make build PROFILE=hf` | découpe + embeddings E5 + index FAISS/BM25 (télécharge ~130 Mo la 1re fois) |
| `make calibrate PROFILE=hf` | calibre les seuils de confiance sur le jeu de **dev** |
| `make eval PROFILE=hf` | mesures sur le jeu de **test** → `results/hf/` (tableaux CSV, figures, JSON) |
| `make examples PROFILE=hf` | conversations d'exemple → `results/hf/conversations_*.json` |
| `make app PROFILE=hf` | interface web Streamlit |
| `make chat PROFILE=hf` | conversation en terminal (`/escalade`, `/quit`) |
| `make report PROFILE=hf REPO_URL=https://github.com/... AUTHORS="Nom Prénom"` | rapport PDF (pdflatex) |
| `make test` | 17 tests (sans réseau) |

Profils : `hf` (référence : `intfloat/e5-small-v2` + cross-encoder `ms-marco-MiniLM-L-6-v2`) et `offline`
(aucun téléchargement : WordLlama + BM25 + re-ranker appris ; c'est avec lui que les résultats du rapport ont été mesurés
dans l'environnement de développement, sans accès à HuggingFace). Sans `make` : `PYTHONPATH=src python -m supportbot <build|calibrate|eval|ask|chat|logs> --profile hf`.

## Brancher un vrai LLM (optionnel, recommandé pour la démo)
```bash
# Ollama (gratuit, local) : https://ollama.com  puis :  ollama pull qwen2.5:3b-instruct
python -m supportbot chat --profile hf --set llm.backend=openai
# ou OpenAI / Groq : --set llm.backend=openai --set llm.base_url=https://api.openai.com/v1 --set llm.model=gpt-4o-mini  (clé dans OPENAI_API_KEY)
# ou modèle local transformers : --set llm.backend=hf-local
```
Dans l'interface, choisissez « openai » dans la barre latérale. Si le LLM tombe en panne, le bot se replie sur une réponse extractive.

## Structure
`src/supportbot/` (ingest, chunking, embeddings, index, reranker, confidence, llm, pipeline, evaluation, logstats) ·
`app/streamlit_app.py` · `configs/{hf,offline}.yaml` · `data/raw` (corpus + manifest) · `data/eval/questions.jsonl` (96 questions étiquetées) ·
`results/` · `tests/` · `report/`.

## Évaluation
96 questions (62 couvertes, 17 hors périmètre, 14 ambiguës, 3 demandes d'humain) ; 1/3 en **dev** (entraînement du re-ranker + calibration
des seuils, validation croisée), 2/3 en **test** (jamais utilisé pour régler). Étiquettes vérifiées contre le corpus (`make validate`).
