from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv


class AppConfig(BaseModel):
    name: str = "PaperWeave"
    environment: str = "development"
    log_level: str = "INFO"
    metrics_file: str = "data/metadata/runtime_metrics.json"


class PathConfig(BaseModel):
    data_dir: str
    raw_pdfs_dir: str
    parsed_text_dir: str
    parsed_markdown_dir: str
    jsonl_dir: str
    metadata_dir: str
    eval_questions_dir: str
    basic_rag_dir: str
    benchmark_dir: str
    evaluation_dir: str
    report_dir: str
    log_dir: str


class PricingConfig(BaseModel):
    prompt_per_1k: float = 0.0
    completion_per_1k: float = 0.0


class ProviderConfig(BaseModel):
    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    pricing: PricingConfig = Field(default_factory=PricingConfig)


class EmbeddingConfig(BaseModel):
    provider: str
    model: str
    base_url: str | None = None
    dimensions: int = 768


class JudgeConfig(BaseModel):
    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None


class ProvidersConfig(BaseModel):
    llm: ProviderConfig
    embeddings: EmbeddingConfig
    judge: JudgeConfig


class DatasetConfig(BaseModel):
    query: str
    max_results: int
    target_tokens: int
    min_tokens_per_paper: int
    request_delay_seconds: float
    categories: list[str]
    curated_ids: list[str] = Field(default_factory=list)
    curated_titles: list[str] = Field(default_factory=list)


class ChunkingConfig(BaseModel):
    strategy: str = "semantic"
    chunk_size: int = 1400
    overlap: int = 200
    semantic_threshold: float = 0.95


class RetrievalConfig(BaseModel):
    basic_rag_top_k: int = 5
    graphrag_top_k: int = 5
    graphrag_num_hops: int = 2
    graphrag_community_level: int = 2


class GraphRAGConfig(BaseModel):
    api_base: str
    graph_name: str
    chunker: str = "semantic"
    extractor: str = "llm"
    top_k: int = 5
    num_hops: int = 2
    community_level: int = 2
    chunk_only: bool = True
    doc_only: bool = False


class TigerGraphConfig(BaseModel):
    hostname: str
    restpp_port: int
    gsql_port: int
    username: str
    password: str
    graph_name: str


class EvaluationConfig(BaseModel):
    bertscore_model: str
    judge_enabled: bool = True
    categories: list[str]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    app_log_level: str = "INFO"
    paperweave_config_path: str = "configs/base.yaml"

    app: AppConfig
    paths: PathConfig
    providers: ProvidersConfig
    dataset: DatasetConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    graphrag: GraphRAGConfig
    tigergraph: TigerGraphConfig
    evaluation: EvaluationConfig

    @classmethod
    def from_yaml(cls) -> "Settings":
        load_dotenv()
        config_path = Path(os.getenv("PAPERWEAVE_CONFIG_PATH", "configs/base.yaml"))
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        raw.setdefault("app", {})
        raw["app"]["environment"] = os.getenv("APP_ENV", raw["app"].get("environment", "development"))
        raw["app"]["log_level"] = os.getenv("APP_LOG_LEVEL", raw["app"].get("log_level", "INFO"))
        raw.setdefault("providers", {})
        raw["providers"].setdefault("llm", {})
        raw["providers"].setdefault("embeddings", {})
        raw["providers"].setdefault("judge", {})
        raw["providers"]["llm"]["provider"] = os.getenv("LLM_PROVIDER", raw["providers"]["llm"].get("provider", "ollama"))
        raw["providers"]["llm"]["model"] = os.getenv("LLM_MODEL", raw["providers"]["llm"].get("model", "qwen2.5:7b"))
        raw["providers"]["llm"]["base_url"] = os.getenv("LLM_BASE_URL", raw["providers"]["llm"].get("base_url"))
        raw["providers"]["embeddings"]["provider"] = os.getenv("EMBEDDING_PROVIDER", raw["providers"]["embeddings"].get("provider", "ollama"))
        raw["providers"]["embeddings"]["model"] = os.getenv("EMBEDDING_MODEL", raw["providers"]["embeddings"].get("model", "nomic-embed-text"))
        raw["providers"]["embeddings"]["base_url"] = os.getenv("EMBEDDING_BASE_URL", raw["providers"]["embeddings"].get("base_url"))
        raw["providers"]["embeddings"]["dimensions"] = int(os.getenv("EMBEDDING_DIMENSIONS", str(raw["providers"]["embeddings"].get("dimensions", 768))))
        raw["providers"]["judge"]["provider"] = os.getenv("JUDGE_PROVIDER", raw["providers"]["judge"].get("provider", "gemini"))
        raw["providers"]["judge"]["model"] = os.getenv("JUDGE_MODEL", raw["providers"]["judge"].get("model", "gemini-2.5-flash"))
        raw["providers"]["judge"]["base_url"] = os.getenv("JUDGE_BASE_URL", raw["providers"]["judge"].get("base_url"))
        raw.setdefault("graphrag", {})
        raw["graphrag"]["api_base"] = os.getenv("GRAPHRAG_API_BASE", raw["graphrag"].get("api_base", "http://localhost:8000"))
        raw["graphrag"]["top_k"] = int(os.getenv("GRAPHRAG_TOP_K", str(raw["graphrag"].get("top_k", 5))))
        raw["graphrag"]["num_hops"] = int(os.getenv("GRAPHRAG_NUM_HOPS", str(raw["graphrag"].get("num_hops", 2))))
        raw["graphrag"]["community_level"] = int(os.getenv("GRAPHRAG_COMMUNITY_LEVEL", str(raw["graphrag"].get("community_level", 2))))
        return cls(**raw)

    def resolve_path(self, value: str) -> Path:
        return Path(value)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_yaml()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
