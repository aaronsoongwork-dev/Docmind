"""Settings read from environment variables (and a .env file if present)."""
import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional; plain environment variables still work
    pass


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_model: str
    llm_fallback_model: str | None
    llm_timeout: float
    embeddings_provider: str
    embeddings_model: str
    reranker_model: str
    chroma_dir: str
    chunk_size: int
    top_k: int      # chunks sent to the LLM
    fetch_k: int    # candidates pulled from each retriever before fusion/rerank
    database_url: str
    agent_max_steps: int


def get_settings() -> Settings:
    e = os.getenv
    return Settings(
        llm_provider=e("LLM_PROVIDER", "gemini"),
        llm_model=e("LLM_MODEL", "gemini-2.5-flash"),
        llm_fallback_model=e("LLM_FALLBACK_MODEL") or None,
        llm_timeout=float(e("LLM_TIMEOUT", "30")),
        embeddings_provider=e("EMBEDDINGS_PROVIDER", "huggingface"),
        embeddings_model=e("EMBEDDINGS_MODEL", "BAAI/bge-small-en-v1.5"),
        reranker_model=e("RERANKER_MODEL", "BAAI/bge-reranker-base"),
        chroma_dir=e("CHROMA_DIR", "./chroma_db"),
        chunk_size=int(e("CHUNK_SIZE", "800")),
        top_k=int(e("TOP_K", "4")),
        fetch_k=int(e("FETCH_K", "15")),
        database_url=e("DATABASE_URL", "sqlite:///./docmind.db"),
        agent_max_steps=int(e("AGENT_MAX_STEPS", "4")),
    )
