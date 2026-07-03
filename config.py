"""
Central, env-driven configuration for the RAG pipeline.
Loaded once and imported wherever settings are needed.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # LLM / Embeddings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Vector store
    vector_store_backend: str = os.getenv("VECTOR_STORE_BACKEND", "chroma")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./.chroma")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "")

    # Redis cache
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    redis_ttl_seconds: int = int(os.getenv("REDIS_TTL_SECONDS", "3600"))

    # Chunking
    chunk_max_tokens: int = int(os.getenv("CHUNK_MAX_TOKENS", "512"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "64"))

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "5"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))


settings = Settings()
