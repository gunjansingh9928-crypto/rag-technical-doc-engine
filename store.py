"""
Vector database integration.

Chroma is used as the default local/self-hosted backend. The interface is
kept intentionally narrow so a Pinecone (or other) backend can be dropped
in without touching pipeline/query code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings
from src.ingestion.chunker import Chunk


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: dict


class VectorStore:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[List[float]]) -> None:
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=list(embeddings),
            documents=[c.text for c in chunks],
            metadatas=[
                {"heading_trail": c.heading_trail, **c.metadata} for c in chunks
            ],
        )

    def query(
        self, query_embedding: List[float], top_k: int | None = None
    ) -> List[RetrievedChunk]:
        top_k = top_k or settings.top_k
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        retrieved: List[RetrievedChunk] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        for chunk_id, text, distance, metadata in zip(ids, docs, distances, metadatas):
            # Chroma returns cosine distance; convert to similarity score.
            similarity = 1 - distance
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    score=similarity,
                    metadata=metadata or {},
                )
            )

        retrieved = [r for r in retrieved if r.score >= settings.similarity_threshold]
        return retrieved

    def count(self) -> int:
        return self._collection.count()
