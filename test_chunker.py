import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.chunker import chunk_document
from src.ingestion.loader import RawDocument


def _make_doc(content: str) -> RawDocument:
    return RawDocument(
        doc_id="test-doc",
        source_path="test.md",
        title="Test Doc",
        content=content,
        metadata={},
    )


def test_code_block_is_never_split():
    code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(200)) + "\n```"
    doc = _make_doc(f"# Heading\n\nSome intro text.\n\n{code}\n\nSome outro text.")

    chunks = chunk_document(doc, max_tokens=50, overlap_tokens=5)

    full_text = "\n".join(c.text for c in chunks)
    assert code in full_text, "Code block must remain intact in at least one chunk"


def test_heading_trail_is_attached():
    doc = _make_doc("# Top\n\n## Sub\n\nSome content under Sub heading.")
    chunks = chunk_document(doc)
    assert any("Sub" in c.heading_trail for c in chunks)


def test_large_paragraph_gets_windowed_with_overlap():
    long_paragraph = " ".join(["word"] * 2000)
    doc = _make_doc(f"# Heading\n\n{long_paragraph}")
    chunks = chunk_document(doc, max_tokens=100, overlap_tokens=20)

    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 100


def test_empty_document_produces_no_chunks():
    doc = _make_doc("")
    chunks = chunk_document(doc)
    assert chunks == []
