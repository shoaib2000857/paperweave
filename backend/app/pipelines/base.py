from __future__ import annotations

import time
from abc import ABC, abstractmethod

from app.models.api import AskRequest, AskResponse, EvaluationResult, RetrievalInfo, SourceRecord, TimingBreakdown, TokenUsage
from app.services.evaluation import EvaluationService


class BasePipeline(ABC):
    pipeline_name: str

    @abstractmethod
    async def run(self, payload: AskRequest) -> AskResponse:
        ...

    async def _with_evaluation(
        self,
        payload: AskRequest,
        answer: str,
        evaluation_service: EvaluationService | None,
    ) -> EvaluationResult | None:
        if not payload.include_evaluation or not evaluation_service:
            return None
        return await evaluation_service.evaluate(payload.question, answer, payload.reference_answer)

    def _build_response(
        self,
        answer: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        sources: list[SourceRecord],
        retrieval_info: RetrievalInfo,
        retrieval_ms: float,
        generation_ms: float,
        evaluation_ms: float = 0.0,
        estimated_cost: float = 0.0,
        evaluation: EvaluationResult | None = None,
        raw: dict | None = None,
    ) -> AskResponse:
        return AskResponse(
            pipeline=self.pipeline_name,
            answer=answer,
            tokens=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            latency=latency_ms,
            estimated_cost=estimated_cost,
            sources=sources,
            retrieval_info=retrieval_info,
            timing_breakdown=TimingBreakdown(
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                evaluation_ms=evaluation_ms,
                total_ms=latency_ms,
            ),
            evaluation=evaluation,
            raw=raw or {},
        )

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int, prompt_per_1k: float, completion_per_1k: float) -> float:
        return (prompt_tokens / 1000.0 * prompt_per_1k) + (completion_tokens / 1000.0 * completion_per_1k)

    def _now_ms(self) -> float:
        return time.perf_counter() * 1000
