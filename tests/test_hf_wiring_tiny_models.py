"""Valide le câblage sentence-transformers / CrossEncoder (mêmes classes que le profil « hf »)
avec de minuscules modèles BERT aléatoires créés localement : aucun téléchargement.
(Ne mesure PAS la qualité des modèles réels, seulement que l'API est correctement utilisée.)"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("sentence_transformers")

from supportbot.reranker import CrossEncoderReranker  # noqa: E402
from supportbot.embeddings import SentenceTransformerEmbedder  # noqa: E402
from supportbot.types import Candidate, Chunk  # noqa: E402


@pytest.fixture(scope="module")
def tiny_dirs(tmp_path_factory):
    from transformers import BertConfig, BertForSequenceClassification, BertModel, BertTokenizerFast
    root = tmp_path_factory.mktemp("tiny")
    words = "how do i install the library with pip gradient accumulation steps cake recipe query passage".split()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
    (root / "vocab.txt").write_text("\n".join(vocab))
    tok = BertTokenizerFast(str(root / "vocab.txt"), do_lower_case=True)
    cfg = dict(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
               intermediate_size=32, max_position_embeddings=64)
    enc, ce = root / "enc", root / "ce"
    BertModel(BertConfig(**cfg)).save_pretrained(enc); tok.save_pretrained(enc)
    BertForSequenceClassification(BertConfig(num_labels=1, **cfg)).save_pretrained(ce); tok.save_pretrained(ce)
    from sentence_transformers import SentenceTransformer
    try:                                                    # chemin des modules selon la version
        from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
    except ImportError:
        from sentence_transformers.models import Pooling, Transformer
    st = SentenceTransformer(modules=[Transformer(str(enc)), Pooling(16, "mean")])
    st.save(str(root / "st"))
    return root / "st", ce


def test_sentence_transformer_embedder_prefixes_and_normalisation(tiny_dirs):
    emb = SentenceTransformerEmbedder(str(tiny_dirs[0]), query_prefix="query: ", passage_prefix="passage: ")
    q = emb.encode_queries(["how do i install the library"])
    p = emb.encode_passages(["how do i install the library", "cake recipe"])
    assert q.shape == (1, 16) and p.shape == (2, 16) and emb.dim == 16
    assert np.allclose(np.linalg.norm(p, axis=1), 1.0, atol=1e-5)
    assert not np.allclose(q, p[:1])           # le préfixe « query: » change bien le vecteur


def test_cross_encoder_reranker_returns_probabilities(tiny_dirs):
    rr = CrossEncoderReranker(str(tiny_dirs[1]), max_length=32)
    mk = lambda i, t: Candidate(Chunk(f"d#{i}", "d", "Doc", "Sec", "u", t, 5), fused_score=1 / (i + 1))
    out = rr.rerank("how do i install", [mk(0, "pip install the library"), mk(1, "cake recipe"), mk(2, "gradient steps")], 2)
    assert len(out) == 2 and all(0.0 < c.rerank_score < 1.0 for c in out)
    assert out[0].rerank_score >= out[1].rerank_score
