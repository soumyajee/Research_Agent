# app/config.py

from functools import lru_cache

from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class Settings(BaseSettings):

    # ========================================================
    # GROQ
    # ========================================================

    groq_api_key: str

    groq_model: str = (
        "openai/gpt-oss-20b"
    )

    groq_fallback_models: str = (
        "openai/gpt-oss-120b"
    )

    # ========================================================
    # LLM
    # ========================================================

    llm_max_completion_tokens: int = 600

    llm_temperature: float = 0.2

    llm_max_retries: int = 2

    llm_retry_base_seconds: float = 2.0

    llm_auto_retry_max_wait_seconds: float = 10.0

    llm_request_timeout_seconds: float = 60.0

    llm_context_max_chars: int = 8000

    # ========================================================
    # ASSEMBLYAI
    # ========================================================

    assemblyai_api_key: str

    assemblyai_language_code: str = "en"

    # ========================================================
    # TTS
    # ========================================================

    tts_rate: int = 175

    tts_volume: float = 1.0

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embedding_dimension: int = 384

    # ========================================================
    # RERANKER
    # ========================================================

    reranker_model: str = (
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # ========================================================
    # NLI
    # ========================================================

    nli_model: str = (
        "cross-encoder/nli-deberta-v3-base"
    )

    # ========================================================
    # DATABASE
    # ========================================================

    database_url: str = (
        "postgresql://postgres:postgres@127.0.0.1:5432/"
        "research_agent"
    )

    # ========================================================
    # REDIS
    # ========================================================

    redis_url: str = (
        "redis://127.0.0.1:6379/0"
    )

    # ========================================================
    # RABBITMQ
    # ========================================================

    rabbitmq_url: str = (
        "amqp://guest:guest@127.0.0.1:5672/"
    )

    # ========================================================
    # RESEARCH
    # ========================================================

    max_retrieval_results: int = 5

    max_web_searches: int = 3

    max_retries: int = 2

    confidence_high: float = 0.85

    confidence_retry: float = 0.65

    # ========================================================
    # STORAGE
    # ========================================================

    object_storage_path: str = "./data"

    backend_url: str = (
        "http://127.0.0.1:8000"
    )

    # ========================================================
    # PYDANTIC
    # ========================================================

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:

    return Settings()


settings = get_settings()