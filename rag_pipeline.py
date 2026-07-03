"""
Orchestrates the ingestion pipeline: load -> chunk -> embed -> index.
"""
from __future__ import annotations

from typing import List

from tqdm import tqdm

from src.cache.redis_cache import RedisCache
from src.embeddings.embedder import Embedder
from src.ingestion.chunker import Chunk, chunk_document
from src.ingestion.loader import iter_documents
from src.vectorstore.store import VectorStore

_BATCH_SIZE = 64


class RAGPipeline:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.embedder = Embedder()
        self.store = VectorStore(collection_name)
        self.cache = RedisCache()

    def build_index(self, source_dir: str) -> int:
        """Ingests every supported doc in `source_dir`, returns chunk count indexed."""
        all_chunks: List[Chunk] = []
        for doc in tqdm(list(iter_documents(source_dir)), desc="loading documents"):
            all_chunks.extend(chunk_document(doc))

        for i in tqdm(range(0, len(all_chunks), _BATCH_SIZE), desc="embedding + indexing"):
            batch = all_chunks[i : i + _BATCH_SIZE]
            embeddings = self.embedder.embed_batch([c.text for c in batch])
            self.store.upsert(batch, embeddings)

        # Any previously cached results are now stale against the new index.
        self.cache.invalidate_collection(self.collection_name)

        return len(all_chunks)
