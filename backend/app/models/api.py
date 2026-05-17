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


class JudgeResult(BaseModel):
    score: float | None = None
    passed: bool | None = None
    reasoning: str | None = None
    hallucination_level: float | None = None
    factual_correctness: float | None = None
    grounding: float | None = None
    completeness: float | None = None
    scientific_accuracy: float | None = None


class HallucinationResult(BaseModel):
    fabricated_citation_count: int = 0
    fabricated_citation_rate: float = 0.0
    answer_context_mismatch: float = 0.0
    unsupported_claim_estimate: float = 0.0
    has_fabricated_citations: bool = False
    high_answer_context_mismatch: bool = False
    high_unsupported_claim_risk: bool = False


class RetrievalQualityResult(BaseModel):
    retrieval_hit: bool = False
    retrieved_chunk_count: int = 0
    source_overlap: float = 0.0
    citation_correctness: float = 0.0
    context_relevance: float = 0.0
    useful_chunk_ratio: float = 0.0
    duplicate_chunk_ratio: float = 0.0


class LivePipelineMetrics(BaseModel):
    total_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    evaluation_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    token_reduction_pct_vs_llm_only: float = 0.0
    bertscore_raw_f1: float | None = None
    bertscore_rescaled_f1: float | None = None
    judge_score: float | None = None
    judge_correctness_pct: float | None = None
    judge_pass: bool | None = None
    retrieval_quality: float = 0.0
    citation_correctness: float = 0.0


class LivePipelineResult(BaseModel):
    answer: str
    tokens: TokenUsage
    latency: float
    estimated_cost: float
    sources: list[SourceRecord]
    retrieval_info: RetrievalInfo
    timing_breakdown: TimingBreakdown
    metrics: LivePipelineMetrics
    judge: JudgeResult
    hallucination: HallucinationResult
    retrieval: RetrievalQualityResult
    evaluation_reference: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


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
    pipelines: dict[str, LivePipelineResult] = Field(default_factory=dict)
    leaderboard: list[dict[str, Any]] = Field(default_factory=list)
    global_metrics: dict[str, Any] = Field(default_factory=dict)
    llm_only: AskResponse | None = None
    basic_rag: AskResponse | None = None
    graphrag: AskResponse | None = None
    errors: dict[str, str] = Field(default_factory=dict)


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
