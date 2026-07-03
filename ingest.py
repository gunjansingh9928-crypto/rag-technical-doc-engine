#!/usr/bin/env python
"""
CLI: build or update the vector index from a folder of technical docs.

Usage:
    python scripts/ingest.py --source ./data/docs --collection technical-docs
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.rag_pipeline import RAGPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest technical docs into the RAG index.")
    parser.add_argument("--source", required=True, help="Path to a folder of documents.")
    parser.add_argument(
        "--collection", default="technical-docs", help="Vector store collection name."
    )
    args = parser.parse_args()

    print(f"Building index '{args.collection}' from '{args.source}'...")
    pipeline = RAGPipeline(collection_name=args.collection)

    start = time.perf_counter()
    chunk_count = pipeline.build_index(args.source)
    elapsed = time.perf_counter() - start

    print(f"Indexed {chunk_count} chunks in {elapsed:.2f}s.")
    print(f"Total vectors in collection: {pipeline.store.count()}")


if __name__ == "__main__":
    main()
