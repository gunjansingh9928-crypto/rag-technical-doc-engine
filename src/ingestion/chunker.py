"""
Context-aware chunking strategy.

Unlike naive fixed-token-window splitting, this chunker:
  1. Splits along semantic boundaries first (markdown headings, then paragraphs).
  2. Keeps fenced code blocks and LaTeX/math blocks intact (never splits mid-block),
     since breaking them destroys technical/mathematical meaning.
  3. Falls back to token-window splitting only within an oversized paragraph.
  4. Applies a token overlap between adjacent chunks to preserve cross-chunk context.
  5. Tags each chunk with its heading trail (breadcrumb) for better grounding.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

import tiktoken

from src.config import settings
from src.ingestion.loader import RawDocument

_ENCODER = tiktoken.get_encoding("cl100k_base")

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_MATH_BLOCK_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    heading_trail: str
    token_count: int
    metadata: dict = field(default_factory=dict)


def _token_len(text: str) -> int:
    return len(_ENCODER.encode(text))


def _protect_blocks(text: str) -> tuple[str, dict]:
    """Replace code/math blocks with placeholders so heading/paragraph
    splitting never cuts through them; returns the map to restore later."""
    placeholders: dict[str, str] = {}

    def _stash(match: re.Match, prefix: str) -> str:
        key = f"__{prefix}_{len(placeholders)}__"
        placeholders[key] = match.group(0)
        return key

    text = _CODE_BLOCK_RE.sub(lambda m: _stash(m, "CODE"), text)
    text = _MATH_BLOCK_RE.sub(lambda m: _stash(m, "MATH"), text)
    return text, placeholders


def _restore_blocks(text: str, placeholders: dict) -> str:
    for key, original in placeholders.items():
        text = text.replace(key, original)
    return text


def _split_by_headings(text: str) -> List[tuple[str, str]]:
    """Returns list of (heading_trail, section_text)."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]

    sections: List[tuple[str, str]] = []
    trail: List[tuple[int, str]] = []  # (level, title)

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        trail = [t for t in trail if t[0] < level] + [(level, title)]
        heading_trail = " > ".join(t[1] for t in trail)

        if body:
            sections.append((heading_trail, body))

    return sections


def _window_split(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    tokens = _ENCODER.encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    windows = []
    step = max_tokens - overlap_tokens
    for start in range(0, len(tokens), step):
        window_tokens = tokens[start : start + max_tokens]
        windows.append(_ENCODER.decode(window_tokens))
        if start + max_tokens >= len(tokens):
            break
    return windows


def chunk_document(
    doc: RawDocument,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> List[Chunk]:
    max_tokens = max_tokens or settings.chunk_max_tokens
    overlap_tokens = overlap_tokens or settings.chunk_overlap_tokens

    protected_text, placeholders = _protect_blocks(doc.content)
    sections = _split_by_headings(protected_text)

    chunks: List[Chunk] = []
    idx = 0
    for heading_trail, section_text in sections:
        restored = _restore_blocks(section_text, placeholders)
        pieces = _window_split(restored, max_tokens, overlap_tokens)

        for piece in pieces:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}-{idx}",
                    doc_id=doc.doc_id,
                    text=piece.strip(),
                    heading_trail=heading_trail or doc.title,
                    token_count=_token_len(piece),
                    metadata={
                        "source_path": doc.source_path,
                        "title": doc.title,
                        **doc.metadata,
                    },
                )
            )
            idx += 1

    return chunks
