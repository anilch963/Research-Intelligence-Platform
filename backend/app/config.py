from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent

RetrieverMode = Literal["vector", "hybrid"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    openai_api_key: SecretStr
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    eval_llm_model: str = "gpt-4o-mini"

    docs_dir: Path = BACKEND_DIR / "data" / "docs"
    chroma_dir: Path = BACKEND_DIR / "data" / "chroma"
    collection_name: str = "documents"

    chunk_size: int = 1000
    chunk_overlap: int = 150

    retriever_mode: RetrieverMode = "vector"
    top_k: int = 4
    bm25_weight: float = 0.5

    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]


settings = Settings()
