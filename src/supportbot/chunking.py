"""Découpage (chunking) des documents en passages.

Stratégie : découpage *structurel* d'abord (titres Markdown), puis *par taille* à l'intérieur
d'une section. On ne coupe jamais au milieu d'un bloc de code (sauf s'il dépasse à lui seul la
taille maximale). Un recouvrement de prose est ajouté entre passages consécutifs d'une même
section pour ne pas perdre le contexte aux frontières.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .types import Chunk, Document

_HEADING = re.compile(r"^(#{1,4})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def slugify(heading: str) -> str:
    """Ancre à la manière de doc-builder : minuscules, tirets, sans ponctuation."""
    s = re.sub(r"[`*_\[\]()]", "", heading.lower())
    s = re.sub(r"[^\w\s-]", "", s)
    return re.sub(r"[\s]+", "-", s.strip())


def count_words(text: str) -> int:
    return len(text.split())


@dataclass
class _Section:
    path: list[str]
    anchor: str
    lines: list[str]


def split_sections(markdown: str) -> list[_Section]:
    """Découpe selon les titres #..#### (en ignorant les # dans les blocs de code)."""
    sections: list[_Section] = [_Section([], "", [])]
    stack: list[tuple[int, str]] = []
    in_fence = False
    for line in markdown.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            level, title = len(m.group(1)), m.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            sections.append(_Section([t for _, t in stack], slugify(title), []))
        else:
            sections[-1].lines.append(line)
    return [s for s in sections if any(l.strip() for l in s.lines)]


def _blocks(lines: list[str]) -> list[tuple[str, str]]:
    """Découpe en blocs ('code' | 'text', contenu) : un bloc de code reste atomique."""
    blocks: list[tuple[str, str]] = []
    buf: list[str] = []
    in_fence = False
    code: list[str] = []

    def flush_text():
        nonlocal buf
        txt = "\n".join(buf).strip()
        if txt:
            blocks.append(("text", txt))
        buf = []

    for line in lines:
        if _FENCE.match(line):
            if not in_fence:
                flush_text()
                in_fence, code = True, [line]
            else:
                code.append(line)
                blocks.append(("code", "\n".join(code)))
                in_fence, code = False, []
            continue
        if in_fence:
            code.append(line)
        elif not line.strip():
            flush_text()          # ligne vide = fin de paragraphe
        else:
            buf.append(line)
    flush_text()
    if code:                       # bloc de code non refermé
        blocks.append(("code", "\n".join(code)))
    return blocks


def _split_oversized(kind: str, text: str, max_words: int) -> list[tuple[str, str]]:
    """Coupe un bloc trop long (par lignes) en morceaux ≤ max_words."""
    if count_words(text) <= max_words:
        return [(kind, text)]
    pieces, cur, n = [], [], 0
    lines: list[str] = []
    for line in text.splitlines():           # une seule ligne énorme (ex. exemple de dataset) → par mots
        if count_words(line) > max_words:
            words = line.split()
            lines.extend(" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words))
        else:
            lines.append(line)
    for line in lines:
        w = count_words(line)
        if cur and n + w > max_words:
            pieces.append((kind, "\n".join(cur)))
            cur, n = [], 0
        cur.append(line)
        n += w
    if cur:
        pieces.append((kind, "\n".join(cur)))
    return pieces


def chunk_document(doc: Document, max_words: int = 220, overlap_words: int = 40,
                   min_words: int = 25) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sec in split_sections(doc.text):
        blocks: list[tuple[str, str]] = []
        for kind, txt in _blocks(sec.lines):
            blocks.extend(_split_oversized(kind, txt, max_words))
        groups: list[list[tuple[str, str]]] = []
        cur: list[tuple[str, str]] = []
        cur_w = 0
        for kind, txt in blocks:
            w = count_words(txt)
            if cur and cur_w + w > max_words:
                groups.append(cur)
                # recouvrement : on reprend les derniers blocs de prose du groupe précédent
                carry, cw = [], 0
                for k2, t2 in reversed(cur):
                    if k2 != "text" or cw + count_words(t2) > overlap_words:
                        break
                    carry.insert(0, (k2, t2))
                    cw += count_words(t2)
                cur, cur_w = carry, cw
            cur.append((kind, txt))
            cur_w += w
        if cur:
            groups.append(cur)
        # fusionne un dernier fragment trop court avec le précédent (de la même section)
        if len(groups) > 1 and sum(count_words(t) for _, t in groups[-1]) < min_words:
            groups[-2].extend(groups.pop())
        for g in groups:
            body = "\n\n".join(t for _, t in g).strip()
            path = sec.path[1:] if len(sec.path) > 1 and sec.path[0] == doc.title else sec.path
            section = "" if path == [doc.title] else " > ".join(path)
            chunks.append(Chunk(
                chunk_id="", doc_id=doc.doc_id, doc_title=doc.title, section=section,
                url=f"{doc.url}#{sec.anchor}" if sec.anchor and len(sec.path) > 1 else doc.url,
                text=body, n_words=count_words(body)))
    # fragments encore trop courts (petites sections isolées) : fusion avec le passage voisin
    merged: list[Chunk] = []
    for c in chunks:
        if merged and (c.n_words < min_words or merged[-1].n_words < min_words) \
                and merged[-1].n_words + c.n_words <= max_words * 1.5:
            prev = merged[-1]
            head = f"{c.section}\n" if c.section and c.section != prev.section else ""
            prev.text = f"{prev.text}\n\n{head}{c.text}".strip()
            prev.n_words = count_words(prev.text)
        else:
            merged.append(c)
    for i, c in enumerate(merged):
        c.chunk_id = f"{doc.doc_id}#{i}"
    chunks = merged
    return chunks


def chunk_corpus(docs: list[Document], max_words: int, overlap_words: int, min_words: int) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, max_words, overlap_words, min_words))
    return out
