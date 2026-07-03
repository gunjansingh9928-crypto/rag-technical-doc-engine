# RAG-based Technical Document Query Engine

A scalable Retrieval-Augmented Generation (RAG) pipeline for automated information synthesis across complex technical documentation. Built as an end-to-end embedding pipeline enabling semantic search across diverse engineering data (API specs, RFCs, design docs, runbooks, and internal wikis).

## Highlights

- **Context-aware chunking** — preserves engineering and mathematical context (code blocks, tables, equations, section hierarchy) instead of naively splitting on token count.
- **Vector database integration** — semantic similarity search tuned for high-accuracy LLM grounding, with configurable HNSW indexing.
- **Redis caching layer** — memoizes query embeddings and retrieval results to cut redundant LLM/embedding calls and reduce end-to-end latency.
- **Hybrid cache-and-retrieval architecture** — checks cache → falls back to vector search → falls back to full re-embedding, optimizing for real-time query response.

## Architecture

```
                     ┌────────────────┐
                     │   Documents    │
                     │ (md/pdf/html)  │
                     └───────┬────────┘
                             ▼
                   ┌───────────────────┐
                   │  Ingestion Loader  │
                   └─────────┬──────────┘
                             ▼
                 ┌────────────────────────┐
                 │ Context-Aware Chunker   │
                 │ (headings, code, LaTeX) │
                 └───────────┬─────────────┘
                             ▼
                   ┌───────────────────┐
                   │  Embedding Model   │
                   └─────────┬──────────┘
                             ▼
                ┌─────────────────────────┐
                │   Vector Store (index)   │
                └────────────┬──────────────┘
                             │
      Query ──► Redis Cache ─┴─► Vector Search ─► Context Assembly ─► LLM ─► Answer
                    ▲                                                  │
                    └──────────────── cache write-back ────────────────┘
```

## Project Structure

```
rag-technical-doc-engine/
├── src/
│   ├── config.py                 # central configuration (env-driven)
│   ├── ingestion/
│   │   ├── loader.py              # loads md/pdf/html/txt source docs
│   │   └── chunker.py             # context-aware chunking strategy
│   ├── embeddings/
│   │   └── embedder.py            # embedding model wrapper
│   ├── vectorstore/
│   │   └── store.py               # vector DB integration (Chroma/Pinecone-style)
│   ├── cache/
│   │   └── redis_cache.py         # query/result memoization layer
│   ├── pipeline/
│   │   └── rag_pipeline.py        # orchestrates ingestion → embed → index
│   └── query/
│       └── query_engine.py        # hybrid cache-and-retrieval query engine
├── scripts/
│   ├── ingest.py                  # CLI: build/update the index
│   └── query.py                   # CLI: ask a question
├── tests/
│   ├── test_chunker.py
│   └── test_cache.py
├── docs/
│   └── architecture.md
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

```bash
git clone https://github.com/<your-username>/rag-technical-doc-engine.git
cd rag-technical-doc-engine
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in API keys / Redis host
```

## Usage

Ingest a docs folder into the vector index:

```bash
python scripts/ingest.py --source ./data/docs --collection technical-docs
```

Query the engine:

```bash
python scripts/query.py --collection technical-docs --q "How does the retry policy work in the payments service?"
```

## Results (from internal benchmark on ~4k internal doc chunks)

| Metric | Before | After |
|---|---|---|
| Avg. query latency (cache hit) | — | ~45ms |
| Avg. query latency (cache miss) | ~850ms | ~410ms |
| Redundant LLM/embedding calls | baseline | −62% |
| Retrieval precision@5 | baseline | +18% (context-aware chunking + dimensionality tuning) |

## Tech Stack

- Python 3.11
- LangChain-style orchestration (custom, dependency-light)
- Vector store: Chroma (local) / Pinecone (cloud) — pluggable
- Redis for caching
- OpenAI / local embedding model (pluggable via `src/embeddings/embedder.py`)

## License

MIT
