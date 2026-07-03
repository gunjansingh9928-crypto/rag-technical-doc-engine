"""
Loads raw technical documentation from disk (markdown, html, pdf, txt)
and normalizes it into a common Document representation before chunking.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from bs4 import BeautifulSoup
from pypdf import PdfReader


@dataclass
class RawDocument:
    doc_id: str
    source_path: str
    title: str
    content: str
    metadata: dict = field(default_factory=dict)


SUPPORTED_EXTENSIONS = {".md", ".markdown", ".html", ".htm", ".pdf", ".txt"}


def _hash_path(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]


def _load_markdown_or_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    return soup.get_text(separator="\n")


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def load_document(path: Path) -> RawDocument:
    ext = path.suffix.lower()
    if ext in {".md", ".markdown", ".txt"}:
        content = _load_markdown_or_text(path)
    elif ext in {".html", ".htm"}:
        content = _load_html(path)
    elif ext == ".pdf":
        content = _load_pdf(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    return RawDocument(
        doc_id=_hash_path(path),
        source_path=str(path),
        title=path.stem,
        content=content,
        metadata={"extension": ext, "size_bytes": path.stat().st_size},
    )


def iter_documents(source_dir: str) -> Iterator[RawDocument]:
    """Recursively walk `source_dir` and yield RawDocuments for supported files."""
    root = Path(source_dir)
    if not root.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                yield load_document(path)
            except Exception as exc:  # noqa: BLE001
                print(f"[loader] skipping {path}: {exc}")
