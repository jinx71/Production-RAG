"""Application settings, loaded from environment variables / .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM (Groq, OpenAI-compatible) ---
    groq_api_key: str = ""
    generation_model: str = "openai/gpt-oss-120b"
    guardrail_model: str = "llama-3.1-8b-instant"

    # --- Embeddings ---
    # "fastembed" -> local ONNX model (no torch); "hashing" -> deterministic, no deps.
    embedding_backend: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384  # only used by the hashing fallback

    # --- Vector store (Chroma, embedded) ---
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "documents"

    # --- Cache (Redis) ---
    redis_url: str = "redis://localhost:6379/0"
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- Retrieval ---
    top_k: int = 5
    candidate_k: int = 20  # pulled from each retriever before fusion
    rrf_k: int = 60  # reciprocal-rank-fusion constant

    # --- Guardrails ---
    relevance_threshold: float = 0.30  # min top dense cosine sim to attempt an answer
    faithfulness_threshold: float = 0.60  # below this, the answer is flagged unverified

    # --- Misc ---
    client_url: str = "http://localhost:5173"
    auto_seed: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
