"""Environment-driven configuration. Secrets are read from the environment only."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    app_provider: str = "openai"
    app_model: str = "gpt-4o-mini"
    judge_provider: str = "ollama"
    judge_model: str = "gpt-oss:20b"
    ollama_base_url: str = "https://ollama.com"
    aws_region: str = "us-east-1"

    embeddings_provider: str = "lexical"
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    agent_prompt_version: str = "v2"
    retrieval_top_k: int = 4

    eval_max_concurrency: int = 2
    eval_num_repetitions: int = 1
    # Judge calls dominate trace volume; disable their tracing to stay under free-tier ingestion limits.
    eval_trace_evaluators: bool = True

    corpus_dir: Path = ROOT / "data" / "corpus"
    prompts_dir: Path = ROOT / "prompts"
    datasets_dir: Path = ROOT / "datasets"
    baselines_dir: Path = ROOT / "baselines"


@lru_cache
def get_settings() -> Settings:
    return Settings()
