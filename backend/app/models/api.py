from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None
    num_hops: int | None = None
    include_evaluation: bool = False
    reference_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceRecord(BaseModel):
    id: str
    title: str | None = None
    snippet: str
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TimingBreakdown(BaseModel):
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    evaluation_ms: float = 0.0
    total_ms: float = 0.0


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class RetrievalInfo(BaseModel):
    mode: str
    top_k: int | None = None
    num_hops: int | None = None
    chunk_strategy: str | None = None
    graph_name: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    bertscore_f1: float | None = None
    judge_pass: bool | None = None
    judge_reasoning: str | None = None


class AskResponse(BaseModel):
    pipeline: str
    answer: str
    tokens: TokenUsage
    latency: float
    estimated_cost: float
    sources: list[SourceRecord]
    retrieval_info: RetrievalInfo
    timing_breakdown: TimingBreakdown
    evaluation: EvaluationResult | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class AskAllResponse(BaseModel):
    question: str
    llm_only: AskResponse
    basic_rag: AskResponse
    graphrag: AskResponse


class BenchmarkQuestion(BaseModel):
    id: str
    category: str
    question: str
    reference_answer: str | None = None


class BenchmarkRequest(BaseModel):
    questions: list[BenchmarkQuestion] = Field(default_factory=list)
    question_file: str | None = None


class BenchmarkResult(BaseModel):
    question_id: str
    category: str
    llm_only: AskResponse
    basic_rag: AskResponse
    graphrag: AskResponse


class BenchmarkResponse(BaseModel):
    results: list[BenchmarkResult]
    summary: dict[str, Any]
