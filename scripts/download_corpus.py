#!/usr/bin/env python3
"""Télécharge le corpus de documentation (Markdown) de HuggingFace Transformers.

Source : fichiers .md de https://github.com/huggingface/transformers/tree/main/docs/source/en
(ce sont exactement les sources des pages https://huggingface.co/docs/transformers/...).

Sortie :
  data/raw/transformers/<page_id avec '/' remplacé par '__'>.md
  data/raw/manifest.json   (URL, taille, sha256, date de téléchargement)

Usage : python scripts/download_corpus.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

RAW_BASE = "https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en"
DOC_BASE = "https://huggingface.co/docs/transformers"

# Pages orientées *utilisateur* (installation, inférence, entraînement, matériel,
# quantization, dépannage). Les pages réservées aux contributeurs sont exclues.
PAGES = """
index installation quicktour
models custom_models model_sharing serialization
fast_tokenizers tokenizer_summary
pipeline_tutorial pipeline_gradio pipeline_webserver
llm_tutorial generation_strategies generation_features tasks/prompting
perf_torch_compile cache_explanation kv_cache assisted_decoding llm_tutorial_optimization
conversations chat_templating chat_extras serve-cli/serving
trainer training trainer_callbacks data_collators optimizers hpo_train trainer_recipes
peft grad_accumulation grad_checkpointing model_memory_anatomy mixed_precision_training
accelerate ddp fsdp deepspeed debugging perf_train_gpu_many
hardware apple_silicon cpu nvidia_gpu amd_gpu
quantization/overview quantization/concept_guide quantization/selecting quantization/save
quantization/bitsandbytes quantization/gptq quantization/awq quantization/gguf
tasks/sequence_classification tasks/token_classification tasks/question_answering
tasks/summarization tasks/translation tasks/language_modeling tasks/asr tasks/image_classification
troubleshooting reference/environment_variables glossary community_integrations/llama_cpp
""".split()


def main() -> int:
    out_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    pages_dir = out_dir / "transformers"
    pages_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"source": RAW_BASE, "fetched_at": datetime.now(timezone.utc).isoformat(), "pages": []}
    failed = []
    for page in PAGES:
        url = f"{RAW_BASE}/{page}.md"
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=30)
                break
            except requests.RequestException as e:  # réseau instable : on réessaie
                r, err = None, e
                time.sleep(1.5)
        if r is None or r.status_code != 200:
            failed.append((page, getattr(r, "status_code", repr(err))))
            continue
        fname = page.replace("/", "__") + ".md"
        (pages_dir / fname).write_bytes(r.content)
        manifest["pages"].append({
            "id": page, "file": f"transformers/{fname}", "url": f"{DOC_BASE}/{page}",
            "source_url": url, "bytes": len(r.content),
            "sha256": hashlib.sha256(r.content).hexdigest(),
        })
        print(f"ok  {page} ({len(r.content)//1024} Ko)")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n{len(manifest['pages'])} pages téléchargées, {len(failed)} échecs")
    for p, s in failed:
        print("ECHEC", p, s)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
