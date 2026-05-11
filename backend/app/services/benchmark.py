from __future__ import annotations

import json
from pathlib import Path

from app.core.config import Settings
from app.models.api import AskRequest, BenchmarkQuestion, BenchmarkRequest, BenchmarkResponse, BenchmarkResult
from app.pipelines.basic_rag import BasicRAGPipeline
from app.pipelines.graphrag import GraphRAGPipeline
from app.pipelines.llm_only import LLMOnlyPipeline
from app.services.evaluation import EvaluationService
from app.services.metrics import MetricsService
from app.storage.benchmark_store import BenchmarkStore


class BenchmarkService:
    def __init__(
        self,
        settings: Settings,
        llm_only_pipeline: LLMOnlyPipeline,
        basic_rag_pipeline: BasicRAGPipeline,
        graphrag_pipeline: GraphRAGPipeline,
        evaluation_service: EvaluationService,
        metrics_service: MetricsService,
        benchmark_store: BenchmarkStore,
    ):
        self.settings = settings
        self.llm_only_pipeline = llm_only_pipeline
        self.basic_rag_pipeline = basic_rag_pipeline
        self.graphrag_pipeline = graphrag_pipeline
        self.evaluation_service = evaluation_service
        self.metrics_service = metrics_service
        self.benchmark_store = benchmark_store

    async def run(self, payload: BenchmarkRequest) -> BenchmarkResponse:
        questions = payload.questions or self._load_questions(payload.question_file)
        results: list[BenchmarkResult] = []
        for question in questions:
            request = AskRequest(
                question=question.question,
            )
            llm_only = await self.llm_only_pipeline.run(request)
            basic_rag = await self.basic_rag_pipeline.run(request)
            graphrag = await self.graphrag_pipeline.run(request)
            if question.reference_answer:
                llm_only.evaluation = await self.evaluation_service.evaluate(question.question, llm_only.answer, question.reference_answer)
                basic_rag.evaluation = await self.evaluation_service.evaluate(question.question, basic_rag.answer, question.reference_answer)
                graphrag.evaluation = await self.evaluation_service.evaluate(question.question, graphrag.answer, question.reference_answer)
            results.append(
                BenchmarkResult(
                    question_id=question.id,
                    category=question.category,
                    llm_only=llm_only,
                    basic_rag=basic_rag,
                    graphrag=graphrag,
                )
            )

        summary = self._summarize(results)
        response = BenchmarkResponse(results=results, summary=summary)
        self.benchmark_store.save(response.model_dump(mode="json"))
        self.metrics_service.record({"type": "benchmark", "summary": summary})
        return response

    def _load_questions(self, question_file: str | None) -> list[BenchmarkQuestion]:
        path = Path(question_file) if question_file else Path(self.settings.paths.eval_questions_dir) / "benchmark_questions.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return [BenchmarkQuestion(**item) for item in data]

    def _summarize(self, results: list[BenchmarkResult]) -> dict:
        def average(items: list[float]) -> float:
            return sum(items) / len(items) if items else 0.0

        return {
            "count": len(results),
            "llm_only_avg_tokens": average([item.llm_only.tokens.total_tokens for item in results]),
            "basic_rag_avg_tokens": average([item.basic_rag.tokens.total_tokens for item in results]),
            "graphrag_avg_tokens": average([item.graphrag.tokens.total_tokens for item in results]),
            "llm_only_avg_latency_ms": average([item.llm_only.latency for item in results]),
            "basic_rag_avg_latency_ms": average([item.basic_rag.latency for item in results]),
            "graphrag_avg_latency_ms": average([item.graphrag.latency for item in results]),
            "llm_only_avg_bertscore": average([item.llm_only.evaluation.bertscore_f1 for item in results if item.llm_only.evaluation and item.llm_only.evaluation.bertscore_f1 is not None]),
            "basic_rag_avg_bertscore": average([item.basic_rag.evaluation.bertscore_f1 for item in results if item.basic_rag.evaluation and item.basic_rag.evaluation.bertscore_f1 is not None]),
            "graphrag_avg_bertscore": average([item.graphrag.evaluation.bertscore_f1 for item in results if item.graphrag.evaluation and item.graphrag.evaluation.bertscore_f1 is not None]),
        }
