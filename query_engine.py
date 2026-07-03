"""
Hybrid cache-and-retrieval query engine.

Flow for every incoming question:
  1. Check Redis for an already-answered identical query (fast path, ~ms).
  2. On miss: check Redis for a cached query embedding; if absent, embed
     the query (this is the expensive call we most want to avoid repeating).
  3. Run vector similarity search against the index.
  4. Assemble retrieved chunks into a grounded context window.
  5. Call the LLM to synthesize an answer.
  6. Write the result back to cache for future identical queries.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

from openai import OpenAI

from src.cache.redis_cache import RedisCache
from src.config import settings
from src.embeddings.embedder import Embedder
from src.vectorstore.store import RetrievedChunk, VectorStore

_SYSTEM_PROMPT = (
    "You are a technical documentation assistant. Answer strictly using the "
    "provided context. If the context does not contain the answer, say so "
    "explicitly rather than guessing. Cite the section/heading you drew from "
    "when relevant."
)

_client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None


@dataclass
class QueryResult:
    answer: str
    sources: List[dict]
    cache_hit: bool
    latency_ms: float
    metadata: dict = field(default_factory=dict)


class QueryEngine:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.embedder = Embedder()
        self.store = VectorStore(collection_name)
        self.cache = RedisCache()

    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        parts = []
        for c in chunks:
            heading = c.metadata.get("heading_trail", "")
            parts.append(f"### {heading}\n{c.text}")
        return "\n\n---\n\n".join(parts)

    def _call_llm(self, question: str, context: str) -> str:
        if _client is None:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Configure .env before querying."
            )
        response = _client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content or ""

    def ask(self, question: str) -> QueryResult:
        start = time.perf_counter()

        # 1. full-result cache
        cached = self.cache.get_result(self.collection_name, question)
        if cached:
            return QueryResult(
                answer=cached["answer"],
                sources=cached["sources"],
                cache_hit=True,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        # 2. embedding cache -> vector search
        embedding = self.cache.get_embedding(question)
        if embedding is None:
            embedding = self.embedder.embed_query(question)
            self.cache.set_embedding(question, embedding)

        retrieved = self.store.query(embedding)
        context = self._build_context(retrieved)

        # 3. LLM synthesis
        answer = self._call_llm(question, context)

        sources = [
            {
                "chunk_id": c.chunk_id,
                "heading_trail": c.metadata.get("heading_trail", ""),
                "source_path": c.metadata.get("source_path", ""),
                "score": round(c.score, 4),
            }
            for c in retrieved
        ]

        payload = {"answer": answer, "sources": sources}
        self.cache.set_result(self.collection_name, question, payload)

        return QueryResult(
            answer=answer,
            sources=sources,
            cache_hit=False,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
