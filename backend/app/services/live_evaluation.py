from __future__ import annotations

import time
from statistics import mean
from typing import Any

from app.core.config import Settings
from app.models.api import (
    AskResponse,
    HallucinationResult,
    JudgeResult,
    LivePipelineMetrics,
    LivePipelineResult,
    RetrievalQualityResult,
)
from app.services.llm import LLMClient
from evaluation.bertscore_eval import evaluate_bertscore
from evaluation.dataset import EvalQuestion
from evaluation.judge import evaluate_with_judge
from evaluation.metrics import (
    add_hallucination_metrics,
    add_retrieval_metrics,
    add_token_metrics,
    response_to_record,
    summarize_pipeline_records,
)


class LiveEvaluationService:
    def __init__(self, settings: Settings, judge_client: LLMClient):
        self.settings = settings
        self.judge_client = judge_client

    async def evaluate_live_query(
        self,
        question: str,
        responses: dict[str, AskResponse],
        reference_answer: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        evaluation_reference = reference_answer.strip() if reference_answer else self._consensus_reference(responses)
        evaluation_reference_source = "user_reference_answer" if reference_answer else "cross_pipeline_consensus"

        eval_question = EvalQuestion(
            id="live-query",
            question=question,
            ground_truth=evaluation_reference,
            category="live",
            difficulty="unknown",
            sources=[],
        )

        records = [response_to_record(eval_question, response) for response in responses.values()]
        add_token_metrics(records, baseline_pipeline="llm-only")
        add_retrieval_metrics(records)
        add_hallucination_metrics(records)

        bert_started = time.perf_counter()
        bertscore_result = evaluate_bertscore(records, model_type=self.settings.evaluation.bertscore_model)
        bertscore_available = True
        bert_ms = (time.perf_counter() - bert_started) * 1000

        judge_started = time.perf_counter()
        judge_result = await evaluate_with_judge(records, self.judge_client)
        judge_ms = (time.perf_counter() - judge_started) * 1000

        per_pipeline_eval_ms = (bert_ms + judge_ms) / max(len(records), 1)
        for record in records:
            record["evaluation_latency_ms"] = per_pipeline_eval_ms

        summary = summarize_pipeline_records(records)
        leaderboard = self._build_leaderboard(summary)
        pipeline_results = self._build_pipeline_results(
            responses=responses,
            records=records,
            evaluation_reference_source=evaluation_reference_source,
            per_pipeline_eval_ms=per_pipeline_eval_ms,
        )

        global_metrics = {
            "question_count": 1,
            "pipeline_count": len(records),
            "avg_total_latency_ms": self._avg([record.get("total_latency_ms") for record in records]),
            "avg_retrieval_latency_ms": self._avg([record.get("retrieval_latency_ms") for record in records]),
            "avg_generation_latency_ms": self._avg([record.get("generation_latency_ms") for record in records]),
            "avg_evaluation_latency_ms": self._avg([record.get("evaluation_latency_ms") for record in records]),
            "avg_total_tokens": self._avg([record.get("total_tokens") for record in records]),
            "best_pipeline": leaderboard[0]["pipeline"] if leaderboard else None,
            "evaluation_reference_source": evaluation_reference_source,
            "evaluation_runtime_ms": (time.perf_counter() - started) * 1000,
        }

        benchmark_like = {
            "dataset": "live-query",
            "question_count": 1,
            "pipelines": list(responses.keys()),
            "top_k": None,
            "records": records,
            "summary": summary,
        }

        return {
            "pipelines": pipeline_results,
            "leaderboard": leaderboard,
            "global_metrics": global_metrics,
            "benchmark": benchmark_like,
            "bertscore": {
                "available": bertscore_available,
                "path": "live://bertscore",
                "data": bertscore_result,
            },
            "judge": {
                "available": True,
                "path": "live://judge",
                "data": judge_result,
            },
            "report": {
                "available": False,
                "path": "live://report",
                "markdown": None,
                "charts": [],
            },
        }

    def _build_leaderboard(self, summary: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        rows = [{"pipeline": pipeline, **values} for pipeline, values in summary.items()]
        rows.sort(key=lambda row: row.get("hackathon_weighted_score", 0.0), reverse=True)
        for index, row in enumerate(rows, start=1):
            row["rank"] = index
        return rows

    def _consensus_reference(self, responses: dict[str, AskResponse]) -> str:
        lines: list[str] = []
        for pipeline in ("llm-only", "basic-rag", "graphrag"):
            response = responses.get(pipeline)
            if not response or not response.answer.strip():
                continue
            lines.append(f"{pipeline}: {response.answer}")
        return "\n\n".join(lines)

    def _build_pipeline_results(
        self,
        responses: dict[str, AskResponse],
        records: list[dict[str, Any]],
        evaluation_reference_source: str,
        per_pipeline_eval_ms: float,
    ) -> dict[str, LivePipelineResult]:
        records_by_pipeline = {record["pipeline"]: record for record in records}
        results: dict[str, LivePipelineResult] = {}

        for pipeline_name, response in responses.items():
            record = records_by_pipeline.get(pipeline_name, {})
            updated_timing = response.timing_breakdown.model_copy(update={"evaluation_ms": per_pipeline_eval_ms})
            results[pipeline_name] = LivePipelineResult(
                answer=response.answer,
                tokens=response.tokens,
                latency=response.latency,
                estimated_cost=response.estimated_cost,
                sources=response.sources,
                retrieval_info=response.retrieval_info,
                timing_breakdown=updated_timing,
                metrics=LivePipelineMetrics(
                    total_latency_ms=float(record.get("total_latency_ms") or response.latency),
                    retrieval_latency_ms=float(record.get("retrieval_latency_ms") or response.timing_breakdown.retrieval_ms),
                    generation_latency_ms=float(record.get("generation_latency_ms") or response.timing_breakdown.generation_ms),
                    evaluation_latency_ms=float(record.get("evaluation_latency_ms") or per_pipeline_eval_ms),
                    prompt_tokens=int(record.get("prompt_tokens") or response.tokens.prompt_tokens),
                    completion_tokens=int(record.get("output_tokens") or response.tokens.completion_tokens),
                    total_tokens=int(record.get("total_tokens") or response.tokens.total_tokens),
                    token_reduction_pct_vs_llm_only=float(record.get("token_reduction_pct_vs_llm_only") or 0.0),
                    bertscore_raw_f1=self._as_optional_float(record.get("bertscore_raw_f1")),
                    bertscore_rescaled_f1=self._as_optional_float(record.get("bertscore_rescaled_f1")),
                    judge_score=self._as_optional_float(record.get("judge_score")),
                    judge_correctness_pct=self._judge_correctness_pct(record),
                    judge_pass=bool(record.get("judge_pass")) if record.get("judge_pass") is not None else None,
                    retrieval_quality=float(record.get("context_relevance") or 0.0),
                    citation_correctness=float(record.get("citation_correctness") or 0.0),
                ),
                judge=JudgeResult(
                    score=self._as_optional_float(record.get("judge_score")),
                    passed=bool(record.get("judge_pass")) if record.get("judge_pass") is not None else None,
                    reasoning=str(record.get("judge_reasoning") or ""),
                    hallucination_level=self._as_optional_float(record.get("judge_hallucination_level")),
                    factual_correctness=self._as_optional_float(record.get("judge_factual_correctness")),
                    grounding=self._as_optional_float(record.get("judge_grounding")),
                    completeness=self._as_optional_float(record.get("judge_completeness")),
                    scientific_accuracy=self._as_optional_float(record.get("judge_scientific_accuracy")),
                ),
                hallucination=HallucinationResult(
                    fabricated_citation_count=int(record.get("fabricated_citation_count") or 0),
                    fabricated_citation_rate=float(record.get("fabricated_citation_rate") or 0.0),
                    answer_context_mismatch=float(record.get("answer_context_mismatch") or 0.0),
                    unsupported_claim_estimate=float(record.get("unsupported_claim_estimate") or 0.0),
                    has_fabricated_citations=bool(record.get("fabricated_citation_count") or 0),
                    high_answer_context_mismatch=float(record.get("answer_context_mismatch") or 0.0) > 0.35,
                    high_unsupported_claim_risk=float(record.get("unsupported_claim_estimate") or 0.0) > 0.5,
                ),
                retrieval=RetrievalQualityResult(
                    retrieval_hit=bool(record.get("retrieval_hit")),
                    retrieved_chunk_count=int(record.get("retrieved_chunk_count") or 0),
                    source_overlap=float(record.get("source_overlap") or 0.0),
                    citation_correctness=float(record.get("citation_correctness") or 0.0),
                    context_relevance=float(record.get("context_relevance") or 0.0),
                    useful_chunk_ratio=float(record.get("useful_chunk_ratio") or 0.0),
                    duplicate_chunk_ratio=float(record.get("duplicate_chunk_ratio") or 0.0),
                ),
                evaluation_reference=evaluation_reference_source,
                raw=response.raw,
            )

        return results

    def _avg(self, values: list[Any]) -> float:
        numeric = [float(value) for value in values if value is not None]
        return float(mean(numeric)) if numeric else 0.0

    def _as_optional_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _judge_correctness_pct(self, record: dict[str, Any]) -> float | None:
        score = self._as_optional_float(record.get("judge_score"))
        if score is None:
            return None
        return max(0.0, min(100.0, (score / 5.0) * 100.0))
