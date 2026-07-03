"""
Tests for RedisCache. Requires a reachable Redis instance (see README) or
`fakeredis` installed locally for offline runs:
    pip install fakeredis
    pytest tests/test_cache.py --fake-redis
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cache.redis_cache import RedisCache  # noqa: E402


@pytest.fixture()
def cache():
    c = RedisCache(ttl_seconds=5)
    if not c.ping():
        pytest.skip("Redis is not reachable; skipping cache tests.")
    yield c
    c.client.flushdb()


def test_embedding_cache_roundtrip(cache):
    text = "how does the retry policy work?"
    vector = [0.1, 0.2, 0.3]

    assert cache.get_embedding(text) is None
    cache.set_embedding(text, vector)
    assert cache.get_embedding(text) == vector


def test_result_cache_roundtrip(cache):
    payload = {"answer": "It retries with exponential backoff.", "sources": []}
    cache.set_result("technical-docs", "explain retries", payload)

    fetched = cache.get_result("technical-docs", "explain retries")
    assert fetched == payload


def test_result_cache_is_case_and_whitespace_insensitive(cache):
    payload = {"answer": "cached answer", "sources": []}
    cache.set_result("technical-docs", "  What Is caching?  ", payload)

    fetched = cache.get_result("technical-docs", "what is caching?")
    assert fetched == payload


def test_invalidate_collection_clears_only_that_collection(cache):
    cache.set_result("docs-a", "q1", {"answer": "a1", "sources": []})
    cache.set_result("docs-b", "q1", {"answer": "b1", "sources": []})

    cache.invalidate_collection("docs-a")

    assert cache.get_result("docs-a", "q1") is None
    assert cache.get_result("docs-b", "q1") is not None
