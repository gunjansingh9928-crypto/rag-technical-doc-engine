#!/usr/bin/env python
"""
CLI: ask a question against an already-built index.

Usage:
    python scripts/query.py --collection technical-docs --q "How does the retry policy work?"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.query.query_engine import QueryEngine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RAG technical doc engine.")
    parser.add_argument(
        "--collection", default="technical-docs", help="Vector store collection name."
    )
    parser.add_argument("--q", required=True, help="Question to ask.")
    args = parser.parse_args()

    engine = QueryEngine(collection_name=args.collection)
    result = engine.ask(args.q)

    print("\n=== Answer ===")
    print(result.answer)

    print("\n=== Sources ===")
    for src in result.sources:
        print(f"- [{src['score']}] {src['heading_trail']} ({src['source_path']})")

    print(f"\n(cache_hit={result.cache_hit}, latency={result.latency_ms:.1f}ms)")


if __name__ == "__main__":
    main()
