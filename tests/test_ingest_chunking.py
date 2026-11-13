from pathlib import Path

from supportbot.chunking import chunk_document, slugify, split_sections
from supportbot.ingest import clean_hf_markdown, html_to_markdown, load_document, load_documents, pdf_to_markdown
from supportbot.textutil import identifiers, tokenize
from supportbot.types import Document

RAW = """<!--Copyright 2024 HF. Licensed under Apache.-->

# Title

<Youtube id="abc"/>

> [!TIP]
> Use this tip.

<hfoptions id="fw">
<hfoption id="pt">

Text for PyTorch.

```py
tok = "<s>"  # <hfoption id="x"> must survive inside code
```

</hfoption>
</hfoptions>

[[autodoc]] Trainer
"""


def test_clean_markdown_removes_noise_but_keeps_code():
    out = clean_hf_markdown(RAW)
    assert "Copyright" not in out and "Youtube" not in out and "[[autodoc]]" not in out
    assert "<hfoptions" not in out and "</hfoption>" not in out
    assert "Tip:" in out and "**Option: pt**" in out and "Text for PyTorch." in out
    assert '<s>' in out and '<hfoption id="x">' in out          # contenu de code intact


def test_html_and_pdf_loaders(tmp_path):
    html = "<html><body><nav>menu</nav><main><h1>Guide</h1><h2>Setup</h2><p>Run it.</p><pre><code>pip install x</code></pre></main></body></html>"
    md = html_to_markdown(html)
    assert "# Guide" in md and "## Setup" in md and "pip install x" in md and "menu" not in md
    from reportlab.pdfgen import canvas
    pdf = tmp_path / "manual.pdf"
    c = canvas.Canvas(str(pdf)); c.drawString(72, 750, "Hello support PDF page one"); c.showPage(); c.save()
    text = pdf_to_markdown(pdf)
    assert "# manual" in text and "## Page 1" in text and "Hello support PDF" in text
    doc = load_document(pdf)
    assert doc.fmt == "pdf" and doc.doc_id == "manual"


def test_load_documents_uses_manifest(tmp_path):
    (tmp_path / "transformers").mkdir()
    (tmp_path / "transformers" / "a__b.md").write_text("# T\n\nbody text here")
    (tmp_path / "manifest.json").write_text('{"pages":[{"id":"a/b","file":"transformers/a__b.md","url":"https://u/a/b"}]}')
    docs = load_documents(tmp_path)
    assert docs[0].doc_id == "a/b" and docs[0].url == "https://u/a/b"


def test_split_sections_ignores_hash_in_code():
    md = "# A\n\ntext\n\n```py\n# not a heading\n```\n\n## B\n\nmore"
    secs = split_sections(md)
    assert [s.path for s in secs] == [["A"], ["A", "B"]]
    assert slugify("Normal Float 4 (NF4)") == "normal-float-4-nf4"


def test_chunking_properties(chunks):
    assert len({c.chunk_id for c in chunks}) == len(chunks)            # ids uniques
    assert all(c.n_words >= 8 for c in chunks)                         # pas de fragment orphelin
    assert any("```py" in c.text and c.text.count("```") % 2 == 0 for c in chunks)   # bloc de code non coupé
    assert any(c.url.endswith("#gradient-accumulation") for c in chunks)             # ancre de section


def test_long_section_is_split_with_overlap():
    para = lambda i: " ".join(f"w{i}_{j}" for j in range(30))
    doc = Document("d", "D", "# D\n\n## S\n\n" + "\n\n".join(para(i) for i in range(6)), "https://u")
    cs = chunk_document(doc, max_words=70, overlap_words=35, min_words=5)
    assert len(cs) >= 3 and all(c.n_words <= 70 + 35 for c in cs)
    assert para(1) in cs[0].text and para(1) in cs[1].text            # recouvrement de prose


def test_tokenize_keeps_identifiers_whole_and_split():
    t = tokenize("Use from_pretrained with TrainingArguments")
    assert "from_pretrained" in t and "pretrain" in t and "trainingarguments" in t
    assert "the" not in tokenize("the model")
    assert "from_pretrained" in identifiers("call from_pretrained() now") or "from_pretrained" in identifiers("call from_pretrained now")
