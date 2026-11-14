"""Collecte et nettoyage de la documentation (Markdown, HTML, PDF, texte).

Chaque format est converti en Markdown « propre » : les titres (#, ##, ###) servent ensuite
au découpage en passages (voir chunking.py), les blocs de code sont conservés tels quels.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .types import Document

DOC_BASE = "https://huggingface.co/docs/transformers"

_FENCE = re.compile(r"^\s*(```|~~~)")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_SELF_CLOSING = re.compile(r"<(Youtube|img|iframe|EditOnGithub|br|hr)\b[^>]*/?>", re.I)
_CLOSING_NOISE = re.compile(r"</(iframe|div|figure|figcaption|summary|details|Youtube)>", re.I)
_OPEN_NOISE = re.compile(r"<(div|figure|figcaption|details|summary)\b[^>]*>", re.I)
_HFOPTION = re.compile(r"<hfoption\b[^>]*\bid=\"([^\"]+)\"[^>]*>", re.I)
_HFOPTIONS = re.compile(r"</?hfoptions\b[^>]*>|</hfoption>", re.I)
_TIP_OPEN = re.compile(r"^\s*<Tip(?:\s+(warning))?>\s*$", re.I)
_TIP_CLOSE = re.compile(r"^\s*</Tip>\s*$", re.I)
_ADMONITION = re.compile(r"^>\s*\[!(\w+)\]\s*$")
_AUTODOC = re.compile(r"^\s*\[\[autodoc\]\].*$")


def clean_hf_markdown(text: str) -> str:
    """Nettoie le Markdown « doc-builder » de HuggingFace (pseudo-MDX).

    - supprime les commentaires HTML (licence), vidéos, images, <div>, etc. ;
    - transforme <hfoption id="pt"> en ligne « Option : pt » (l'information est conservée) ;
    - transforme <Tip> et `> [!TIP]` en préfixe « Tip: » / « Warning: » ;
    - NE TOUCHE PAS au contenu des blocs de code (``` … ```), où des balises comme <s> ou <eos>
      sont du contenu légitime.
    """
    text = _HTML_COMMENT.sub("", text)
    out: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            out.append(line.rstrip())
            continue
        if in_fence:
            out.append(line.rstrip())
            continue
        if _AUTODOC.match(line):
            continue
        m = _TIP_OPEN.match(line)
        if m:
            out.append("Warning:" if m.group(1) else "Tip:")
            continue
        if _TIP_CLOSE.match(line):
            continue
        m = _ADMONITION.match(line)
        if m:
            out.append(f"{m.group(1).capitalize()}:")
            continue
        line = re.sub(r"^>\s?", "", line) if line.startswith(">") else line
        line = _HFOPTION.sub(lambda mm: f"\n**Option: {mm.group(1)}**\n", line)
        line = _HFOPTIONS.sub("", line)
        line = _SELF_CLOSING.sub("", line)
        line = _CLOSING_NOISE.sub("", line)
        line = _OPEN_NOISE.sub("", line)
        out.append(line.rstrip())
    cleaned = "\n".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() + "\n"


def title_of(markdown: str, default: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return default


def html_to_markdown(html: str) -> str:
    """HTML → Markdown (titres, listes, blocs de code conservés)."""
    from bs4 import BeautifulSoup
    from markdownify import markdownify

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    md = markdownify(str(root), heading_style="ATX", code_language="")
    return re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"


def pdf_to_markdown(path: Path) -> str:
    """PDF → texte. Pas de titres détectables de façon fiable : une section « Page n » par page."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts = []
    for i, page in enumerate(reader.pages, 1):
        txt = (page.extract_text() or "").strip()
        if txt:
            parts.append(f"## Page {i}\n\n{txt}")
    return f"# {path.stem}\n\n" + "\n\n".join(parts) + "\n"


def load_document(path: Path, doc_id: str | None = None, url: str | None = None) -> Document:
    suffix = path.suffix.lower()
    doc_id = doc_id or path.stem
    if suffix in {".md", ".markdown"}:
        text, fmt = clean_hf_markdown(path.read_text(encoding="utf-8")), "md"
    elif suffix in {".html", ".htm"}:
        text, fmt = html_to_markdown(path.read_text(encoding="utf-8")), "html"
    elif suffix == ".pdf":
        text, fmt = pdf_to_markdown(path), "pdf"
    elif suffix == ".txt":
        text, fmt = f"# {path.stem}\n\n" + path.read_text(encoding="utf-8"), "txt"
    else:
        raise ValueError(f"Format non supporté : {path}")
    return Document(doc_id=doc_id, title=title_of(text, doc_id), text=text,
                    url=url or f"{DOC_BASE}/{doc_id}", fmt=fmt)


def load_documents(raw_dir: Path) -> list[Document]:
    """Charge tous les documents de `raw_dir` (récursif). Le manifest, s'il existe, fournit
    l'identifiant de page et l'URL publique des pages HuggingFace."""
    manifest_path = raw_dir / "manifest.json"
    by_file: dict[str, dict] = {}
    if manifest_path.exists():
        for p in json.loads(manifest_path.read_text(encoding="utf-8"))["pages"]:
            by_file[p["file"]] = p
    docs: list[Document] = []
    for path in sorted(raw_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".markdown", ".html", ".htm", ".pdf", ".txt"}:
            continue
        rel = path.relative_to(raw_dir).as_posix()
        meta = by_file.get(rel)
        docs.append(load_document(path, doc_id=meta["id"] if meta else None,
                                  url=meta["url"] if meta else None))
    return docs
