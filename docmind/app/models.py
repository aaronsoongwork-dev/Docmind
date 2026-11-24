"""Model factories. Swapping providers is a config change, not a code change."""
import os
from functools import lru_cache

from .config import get_settings


def get_embeddings(model_name: str | None = None):
    s = get_settings()
    if s.embeddings_provider == "fake":  # tests only: deterministic, not semantic
        from langchain_core.embeddings import DeterministicFakeEmbedding
        return DeterministicFakeEmbedding(size=256)
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=model_name or s.embeddings_model,   # a HF id OR a local fine-tuned folder
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity = dot product
    )


def _build_chat_model(provider: str, model: str, timeout: float):
    """temperature=0 (consistent, fewer hallucinations); retries happen in LCEL, not in SDKs."""
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=0, timeout=timeout, max_retries=0)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0, timeout=timeout, max_retries=0)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=0, timeout=timeout, max_retries=0)
    if provider == "ollama":  # fully local model hosting
        from langchain_ollama import ChatOllama
        kwargs = {"base_url": os.environ["OLLAMA_BASE_URL"]} if os.getenv("OLLAMA_BASE_URL") else {}
        return ChatOllama(model=model, temperature=0, client_kwargs={"timeout": timeout}, **kwargs)
    raise ValueError(f"Unknown LLM provider: {provider}")


def get_llm(provider: str | None = None, model: str | None = None):
    """Defaults come from settings; benchmarks pass explicit provider/model."""
    s = get_settings()
    provider = provider or s.llm_provider
    llm = _build_chat_model(provider, model or s.llm_model, s.llm_timeout)
    # Optional fallback (same provider) for overloaded-model errors such as Gemini 503.
    if model is None and s.llm_fallback_model:
        llm = llm.with_fallbacks([_build_chat_model(provider, s.llm_fallback_model, s.llm_timeout)])
    return llm


@lru_cache(maxsize=1)
def get_reranker():
    """Cross-encoder: reads (query, chunk) TOGETHER, so it is slower but more accurate than embeddings."""
    s = get_settings()
    if s.reranker_model.lower() == "none":
        return None
    from sentence_transformers import CrossEncoder
    return CrossEncoder(s.reranker_model)
