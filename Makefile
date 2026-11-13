PROFILE ?= hf
LLM ?= extractive
export PYTHONPATH := src

setup:            ## installe les dépendances
	pip install -r requirements.txt
corpus:           ## télécharge la documentation (68 pages Markdown)
	python scripts/download_corpus.py
build:            ## découpe + embeddings + index FAISS/BM25
	python -m supportbot build --profile $(PROFILE)
validate:         ## vérifie les étiquettes du jeu de test contre le corpus
	python scripts/validate_eval_set.py
calibrate:        ## entraîne le re-ranker (offline) et calibre les seuils sur le DEV
	python -m supportbot calibrate --profile $(PROFILE)
eval:             ## évalue sur le TEST -> results/$(PROFILE)/
	python -m supportbot eval --profile $(PROFILE) --split test
examples:         ## conversations d'exemple pour le rapport
	python scripts/make_examples.py $(PROFILE) $(LLM)
test:             ## tests unitaires (sans réseau)
	python -m pytest -q
app:              ## interface web
	SUPPORTBOT_PROFILE=$(PROFILE) streamlit run app/streamlit_app.py
chat:             ## conversation en terminal
	python -m supportbot chat --profile $(PROFILE)
report:           ## rapport PDF (nécessite pdflatex) ; REPO_URL=https://github.com/... AUTHORS="Nom Prénom"
	python scripts/build_report.py $(PROFILE)
all: corpus build calibrate eval examples report
