# Architecture Deep Dive

## 1. Ingestion & Context-Aware Chunking

Naive RAG pipelines split documents into fixed-size token windows. For
technical documentation this is destructive: it can slice a code sample in
half, separate a table from its caption, or break a LaTeX equation across two
chunks — all of which destroys retrieval quality and confuses the LLM at
answer time.

`src/ingestion/chunker.py` addresses this with a layered strategy:

1. **Protect atomic blocks.** Fenced code blocks (```` ``` ````) and LaTeX
   math blocks (`$$...$$`) are extracted and replaced with placeholders
   before any splitting happens, guaranteeing they're never cut.
2. **Split on heading structure.** Markdown headings (`#`–`######`) define
   the primary chunk boundaries, and each chunk is tagged with its full
   heading trail (e.g. `API Reference > Authentication > Token Refresh`) so
   retrieved chunks carry their document context even in isolation.
3. **Token-window fallback.** Oversized sections are further split using a
   `tiktoken`-based sliding window with configurable overlap
   (`CHUNK_MAX_TOKENS`, `CHUNK_OVERLAP_TOKENS`), preserving continuity across
   chunk boundaries.
4. **Restore atomic blocks** before finalizing chunk text.

## 2. Embedding & Vector Store

`src/embeddings/embedder.py` wraps the embedding provider behind a stable
interface (`embed_batch`, `embed_query`), so swapping providers (OpenAI,
Cohere, a local sentence-transformer) doesn't ripple through the pipeline.

`src/vectorstore/store.py` wraps Chroma with cosine similarity search and a
configurable `SIMILARITY_THRESHOLD` that filters out low-confidence matches
before they ever reach the LLM — this is a large part of what drives
retrieval precision, alongside chunk quality.

## 3. Redis Caching Layer

Two independent caches live in `src/cache/redis_cache.py`:

- **Embedding cache** — keyed on a normalized hash of the query text. Avoids
  re-embedding identical or near-duplicate queries (common in interactive
  usage where users refine a question incrementally).
- **Result cache** — keyed on `(collection, query)`, storing the full
  answer + source payload. Serves repeat questions in single-digit
  milliseconds without touching the vector store or the LLM at all.

Both use a TTL (`REDIS_TTL_SECONDS`) so answers don't go stale indefinitely,
and `invalidate_collection()` is called automatically after re-ingestion so a
newly updated index is never masked by cached answers from a previous
version of the docs.

## 4. Hybrid Cache-and-Retrieval Query Flow

```
question
   │
   ▼
[result cache hit?] ──yes──► return cached answer (fast path)
   │no
   ▼
[embedding cache hit?] ──no──► embed query ──► cache embedding
   │yes
   ▼
vector similarity search (top_k, threshold-filtered)
   │
   ▼
assemble context (heading-tagged chunks)
   │
   ▼
LLM synthesis
   │
   ▼
cache result ──► return answer + sources
```

This staged fallback is what the résumé bullet calls the "hybrid
cache-and-retrieval architecture" — every stage is optimized to avoid the
most expensive operation (LLM synthesis) first, then the second most
expensive (embedding), only doing full work on a genuine cache miss.

## 5. Why Redis specifically

- Sub-millisecond GET/SET at the scale of a few thousand cached
  queries/embeddings.
- TTL support built in, avoiding a separate cache-eviction cron job.
- Easy to run alongside the app in Docker Compose or as a managed service in
  production without adding a new class of infrastructure (e.g. no need for
  a separate memcached/Postgres solely for caching).
