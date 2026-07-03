"""
Thin wrapper around the embedding provider so the rest of the pipeline
doesn't care whether embeddings come from OpenAI, a local model, etc.
"""
from __future__ import annotations

from typing import List

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

_client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None


class Embedder:
    def __init__(self, model: str | None = None):
        self.model = model or settings.embedding_model

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(4))
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if _client is None:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Configure .env before generating embeddings."
            )
        response = _client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]
