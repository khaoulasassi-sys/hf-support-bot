import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from supportbot.chunking import chunk_corpus  # noqa: E402
from supportbot.types import Document  # noqa: E402


class FakeEmbedder:
    """Embeddings déterministes par « sac de mots haché » : pas de modèle, pas de réseau."""
    name, dim = "fake-bow", 128

    def _enc(self, texts):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9_]+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(n, 1e-9, None)

    encode_passages = encode_queries = lambda self, texts: self._enc(texts)


DOCS = [
    Document("install", "Installation", "# Installation\n\n## Virtual environment\n\nCreate a virtual environment with `python -m venv .env` "
             "then activate it. Install the library with pip install transformers inside the virtual environment to keep "
             "your system clean and avoid dependency conflicts between projects.\n\n## Offline mode\n\nSet HF_HUB_OFFLINE=1 to run "
             "Transformers without internet access, using only files already stored in the local cache directory.", "https://x/install"),
    Document("training", "Training", "# Training\n\n## Gradient accumulation\n\nSet `gradient_accumulation_steps` in TrainingArguments "
             "to simulate a larger batch size when the GPU memory is limited. The optimizer step happens every N batches.\n\n"
             "```py\nargs = TrainingArguments(gradient_accumulation_steps=4)\n```\n\n## Early stopping\n\nUse EarlyStoppingCallback to "
             "stop training when the evaluation metric stops improving for several evaluations in a row.", "https://x/training"),
    Document("quant", "Quantization", "# Quantization\n\n## 4-bit loading\n\nPass `load_in_4bit=True` inside BitsAndBytesConfig to load "
             "a large language model with four bits per weight which divides the memory footprint by roughly four.", "https://x/quant"),
]


@pytest.fixture(scope="session")
def chunks():
    return chunk_corpus(DOCS, max_words=60, overlap_words=10, min_words=8)


@pytest.fixture()
def fake_embedder():
    return FakeEmbedder()
