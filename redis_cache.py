"""
Redis caching layer.

Memoizes two things to cut redundant LLM/embedding inference:
  1. Query embeddings (query text -> vector), so repeated/similar queries
     don't re-hit the embedding API.
  2. Full retrieval + answer results (query hash -> answer payload), so an
     identical question against an unchanged index is served instantly.

This is the backbone of the hybrid cache-and-retrieval architecture:
cache lookup -> vector search on miss -> cache write-back.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

import redis

from src.config import settings

_pool = redis.ConnectionPool(
    host=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    decode_responses=True,
)


def _key(namespace: str, raw: str) -> str:
    digest = hashlib.sha256(raw.strip().lower().encode("utf-8")).hexdigest()
    return f"rag:{namespace}:{digest}"


class RedisCache:
    def __init__(self, ttl_seconds: int | None = None):
        self.client = redis.Redis(connection_pool=_pool)
        self.ttl_seconds = ttl_seconds or settings.redis_ttl_seconds

    # -- embedding cache -------------------------------------------------
    def get_embedding(self, text: str) -> Optional[list[float]]:
        raw = self.client.get(_key("embedding", text))
        return json.loads(raw) if raw else None

    def set_embedding(self, text: str, embedding: list[float]) -> None:
        self.client.set(_key("embedding", text), json.dumps(embedding), ex=self.ttl_seconds)

    # -- query/answer result cache ---------------------------------------
    def get_result(self, collection: str, query: str) -> Optional[dict[str, Any]]:
        raw = self.client.get(_key(f"result:{collection}", query))
        return json.loads(raw) if raw else None

    def set_result(self, collection: str, query: str, payload: dict[str, Any]) -> None:
        self.client.set(
            _key(f"result:{collection}", query), json.dumps(payload), ex=self.ttl_seconds
        )

    # -- utility -----------------------------------------------------------
    def invalidate_collection(self, collection: str) -> int:
        """Best-effort cache bust for a collection after re-ingestion."""
        pattern = _key(f"result:{collection}", "*").replace(
            _key(f"result:{collection}", "")[:-64], f"rag:result:{collection}:*"
        )
        keys = list(self.client.scan_iter(match=f"rag:result:{collection}:*"))
        if keys:
            return self.client.delete(*keys)
        return 0

    def ping(self) -> bool:
        try:
            return self.client.ping()
        except redis.exceptions.ConnectionError:
            return False
