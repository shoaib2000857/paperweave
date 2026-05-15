from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings, get_settings
from app.pipelines.basic_rag import BasicRAGPipeline
from app.pipelines.graphrag import GraphRAGPipeline
from app.pipelines.llm_only import LLMOnlyPipeline
from app.services.benchmark import BenchmarkService
from app.services.evaluation import EvaluationService
from app.services.live_evaluation import LiveEvaluationService
from app.services.llm import LLMClient
from app.services.metrics import MetricsService
from app.services.providers import EmbeddingProvider, LLMProviderFactory
from app.storage.benchmark_store import BenchmarkStore
from app.storage.dataset_store import DatasetStore
from app.storage.rag_store import BasicRAGStore


@dataclass
class Container:
    settings: Settings
    llm_client: LLMClient
    judge_client: LLMClient
    embedding_provider: EmbeddingProvider
    dataset_store: DatasetStore
    rag_store: BasicRAGStore
    metrics_service: MetricsService
    evaluation_service: EvaluationService
    live_evaluation_service: LiveEvaluationService
    llm_only_pipeline: LLMOnlyPipeline
    basic_rag_pipeline: BasicRAGPipeline
    graphrag_pipeline: GraphRAGPipeline
    benchmark_service: BenchmarkService


def build_container() -> Container:
    settings = get_settings()
    llm_provider = LLMProviderFactory(settings).build_llm_provider()
    judge_provider = LLMProviderFactory(settings).build_judge_provider()
    embedding_provider = LLMProviderFactory(settings).build_embedding_provider()
    llm_client = LLMClient(settings=settings, provider=llm_provider)
    judge_client = LLMClient(settings=settings, provider=judge_provider)
    dataset_store = DatasetStore(settings)
    rag_store = BasicRAGStore(settings)
    metrics_service = MetricsService(settings)
    evaluation_service = EvaluationService(settings=settings, llm_client=judge_client)
    live_evaluation_service = LiveEvaluationService(settings=settings, judge_client=judge_client)
    llm_only_pipeline = LLMOnlyPipeline(settings=settings, llm_client=llm_client)
    basic_rag_pipeline = BasicRAGPipeline(
        settings=settings,
        llm_client=llm_client,
        embedding_provider=embedding_provider,
        rag_store=rag_store,
        dataset_store=dataset_store,
    )
    graphrag_pipeline = GraphRAGPipeline(settings=settings, llm_client=llm_client)
    benchmark_store = BenchmarkStore(settings)
    benchmark_service = BenchmarkService(
        settings=settings,
        llm_only_pipeline=llm_only_pipeline,
        basic_rag_pipeline=basic_rag_pipeline,
        graphrag_pipeline=graphrag_pipeline,
        evaluation_service=evaluation_service,
        metrics_service=metrics_service,
        benchmark_store=benchmark_store,
    )
    return Container(
        settings=settings,
        llm_client=llm_client,
        judge_client=judge_client,
        embedding_provider=embedding_provider,
        dataset_store=dataset_store,
        rag_store=rag_store,
        metrics_service=metrics_service,
        evaluation_service=evaluation_service,
        live_evaluation_service=live_evaluation_service,
        llm_only_pipeline=llm_only_pipeline,
        basic_rag_pipeline=basic_rag_pipeline,
        graphrag_pipeline=graphrag_pipeline,
        benchmark_service=benchmark_service,
    )
