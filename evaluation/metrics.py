from __future__ import annotations

import math
import re
from collections import Counter
from statistics import mean
from typing import Any

import numpy as np

from app.models.api import AskResponse
from app.utils.tokens import count_tokens
from evaluation.dataset import EvalQuestion


PIPELINE_WEIGHTS = {
    "token_reduction": 0.30,
    "answer_accuracy": 0.30,
    "performance_latency": 0.20,
    "engineering_storytelling": 0.20,
}


def safe_mean(values: list[float | int | None]) -> float:
    numeric = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return float(mean(numeric)) if numeric else 0.0


def percentile(values: list[float | int | None], q: float) -> float:
    numeric = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not numeric:
        return 0.0
    return float(np.percentile(numeric, q))


def response_to_record(question: EvalQuestion, response: AskResponse, error: str | None = None) -> dict[str, Any]:
    sources = [source.model_dump(mode="json") for source in response.sources]
    context = "\n\n".join(source.get("snippet", "") for source in sources)
    context_tokens = count_tokens(context) if context else 0
    source_titles = [
        str(source.get("title") or source.get("id") or source.get("metadata", {}).get("source") or "")
        for source in sources
    ]
    return {
        "question_id": question.id,
        "question": question.question,
        "ground_truth": question.ground_truth,
        "category": question.category,
        "difficulty": question.difficulty,
        "expected_sources": question.sources,
        "pipeline": response.pipeline,
        "answer": response.answer,
        "retrieved_context": context,
        "sources": sources,
        "source_titles": source_titles,
        "prompt_tokens": response.tokens.prompt_tokens,
        "retrieved_context_tokens": context_tokens,
        "output_tokens": response.tokens.completion_tokens,
        "total_tokens": response.tokens.total_tokens,
        "retrieval_latency_ms": response.timing_breakdown.retrieval_ms,
        "generation_latency_ms": response.timing_breakdown.generation_ms,
        "evaluation_latency_ms": response.timing_breakdown.evaluation_ms,
        "total_latency_ms": response.timing_breakdown.total_ms or response.latency,
        "retrieval_mode": response.retrieval_info.mode,
        "retrieved_chunk_count": len(sources),
        "duplicate_chunk_ratio": duplicate_ratio([source.get("snippet", "") for source in sources]),
        "raw": response.raw,
        "error": error,
    }


def failed_record(question: EvalQuestion, pipeline: str, error: str, latency_ms: float) -> dict[str, Any]:
    return {
        "question_id": question.id,
        "question": question.question,
        "ground_truth": question.ground_truth,
        "category": question.category,
        "difficulty": question.difficulty,
        "expected_sources": question.sources,
        "pipeline": pipeline,
        "answer": "",
        "retrieved_context": "",
        "sources": [],
        "source_titles": [],
        "prompt_tokens": 0,
        "retrieved_context_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "retrieval_latency_ms": 0.0,
        "generation_latency_ms": 0.0,
        "evaluation_latency_ms": 0.0,
        "total_latency_ms": latency_ms,
        "retrieval_mode": "failed",
        "retrieved_chunk_count": 0,
        "duplicate_chunk_ratio": 0.0,
        "raw": {},
        "error": error,
    }


def add_token_metrics(records: list[dict[str, Any]], baseline_pipeline: str = "llm-only") -> None:
    baselines = {
        record["question_id"]: max(float(record.get("total_tokens") or 0), 1.0)
        for record in records
        if record["pipeline"] == baseline_pipeline
    }
    for record in records:
        baseline_tokens = baselines.get(record["question_id"], max(float(record.get("total_tokens") or 0), 1.0))
        total_tokens = float(record.get("total_tokens") or 0)
        prompt_tokens = float(record.get("prompt_tokens") or 0)
        output_tokens = float(record.get("output_tokens") or 0)
        context_tokens = float(record.get("retrieved_context_tokens") or 0)
        if record["pipeline"] == baseline_pipeline:
            record["token_reduction_pct_vs_llm_only"] = 0.0
        else:
            record["token_reduction_pct_vs_llm_only"] = ((baseline_tokens - total_tokens) / baseline_tokens) * 100.0
        record["answer_token_efficiency"] = len(str(record.get("answer") or "").split()) / max(total_tokens, 1.0)
        record["retrieval_compression_efficiency"] = output_tokens / max(context_tokens, 1.0) if context_tokens else 0.0
        record["prompt_context_share"] = context_tokens / max(prompt_tokens, 1.0)


def add_retrieval_metrics(records: list[dict[str, Any]]) -> None:
    for record in records:
        expected = normalize_source_set(record.get("expected_sources") or [])
        retrieved = normalize_source_set(record.get("source_titles") or [])
        context = str(record.get("retrieved_context") or "")
        answer = str(record.get("answer") or "")
        overlap = expected & retrieved if expected else set()
        record["source_overlap"] = len(overlap) / len(expected) if expected else 0.0
        record["retrieval_hit"] = bool(overlap) if expected else bool(record.get("retrieved_chunk_count", 0))
        record["citation_correctness"] = citation_correctness(answer, record.get("sources") or [])
        record["context_relevance"] = lexical_overlap(record.get("question", ""), context)
        record["useful_chunk_ratio"] = useful_chunk_ratio(record.get("question", ""), record.get("sources") or [])


def add_hallucination_metrics(records: list[dict[str, Any]]) -> None:
    for record in records:
        answer = str(record.get("answer") or "")
        context = str(record.get("retrieved_context") or "")
        citations = extract_citations(answer)
        available_citations = {str(index) for index in range(1, int(record.get("retrieved_chunk_count") or 0) + 1)}
        fabricated = [citation for citation in citations if citation not in available_citations]
        answer_terms = content_terms(answer)
        context_terms = content_terms(context)
        unsupported_terms = answer_terms - context_terms if context_terms else set()
        record["fabricated_citation_count"] = len(fabricated)
        record["fabricated_citation_rate"] = len(fabricated) / max(len(citations), 1)
        record["answer_context_mismatch"] = len(unsupported_terms) / max(len(answer_terms), 1) if answer_terms else 0.0
        record["unsupported_claim_estimate"] = unsupported_claim_estimate(answer, context)


def summarize_pipeline_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    pipelines = sorted({record["pipeline"] for record in records})
    for pipeline in pipelines:
        items = [record for record in records if record["pipeline"] == pipeline]
        summary[pipeline] = {
            "count": len(items),
            "failures": sum(1 for item in items if item.get("error")),
            "avg_prompt_tokens": safe_mean([item.get("prompt_tokens") for item in items]),
            "avg_retrieved_context_tokens": safe_mean([item.get("retrieved_context_tokens") for item in items]),
            "avg_output_tokens": safe_mean([item.get("output_tokens") for item in items]),
            "avg_total_tokens": safe_mean([item.get("total_tokens") for item in items]),
            "avg_token_reduction_pct_vs_llm_only": safe_mean([item.get("token_reduction_pct_vs_llm_only") for item in items]),
            "avg_answer_token_efficiency": safe_mean([item.get("answer_token_efficiency") for item in items]),
            "avg_retrieval_compression_efficiency": safe_mean([item.get("retrieval_compression_efficiency") for item in items]),
            "avg_retrieval_latency_ms": safe_mean([item.get("retrieval_latency_ms") for item in items]),
            "avg_generation_latency_ms": safe_mean([item.get("generation_latency_ms") for item in items]),
            "avg_total_latency_ms": safe_mean([item.get("total_latency_ms") for item in items]),
            "p50_total_latency_ms": percentile([item.get("total_latency_ms") for item in items], 50),
            "p95_total_latency_ms": percentile([item.get("total_latency_ms") for item in items], 95),
            "retrieval_hit_rate": safe_mean([1.0 if item.get("retrieval_hit") else 0.0 for item in items]),
            "avg_retrieved_chunk_count": safe_mean([item.get("retrieved_chunk_count") for item in items]),
            "avg_source_overlap": safe_mean([item.get("source_overlap") for item in items]),
            "avg_citation_correctness": safe_mean([item.get("citation_correctness") for item in items]),
            "avg_context_relevance": safe_mean([item.get("context_relevance") for item in items]),
            "avg_useful_chunk_ratio": safe_mean([item.get("useful_chunk_ratio") for item in items]),
            "avg_duplicate_chunk_ratio": safe_mean([item.get("duplicate_chunk_ratio") for item in items]),
            "avg_fabricated_citation_rate": safe_mean([item.get("fabricated_citation_rate") for item in items]),
            "avg_answer_context_mismatch": safe_mean([item.get("answer_context_mismatch") for item in items]),
            "avg_unsupported_claim_estimate": safe_mean([item.get("unsupported_claim_estimate") for item in items]),
            "avg_bertscore_raw_f1": safe_mean([item.get("bertscore_raw_f1") for item in items]),
            "avg_bertscore_rescaled_f1": safe_mean([item.get("bertscore_rescaled_f1") for item in items]),
            "judge_pass_rate": safe_mean([1.0 if item.get("judge_pass") else 0.0 for item in items if item.get("judge_score") is not None]),
            "avg_judge_score": safe_mean([item.get("judge_score") for item in items]),
            "avg_judge_correctness_pct": safe_mean([item.get("judge_correctness_pct") for item in items]),
            "avg_judge_grounding": safe_mean([item.get("judge_grounding") for item in items]),
            "avg_judge_factual_correctness": safe_mean([item.get("judge_factual_correctness") for item in items]),
            "avg_judge_completeness": safe_mean([item.get("judge_completeness") for item in items]),
            "avg_judge_scientific_accuracy": safe_mean([item.get("judge_scientific_accuracy") for item in items]),
            "judge_hallucination_rate": safe_mean([item.get("judge_hallucination_level") for item in items]),
        }
    add_leaderboard_scores(summary)
    return summary


def add_leaderboard_scores(summary: dict[str, dict[str, Any]]) -> None:
    max_tokens = max((data["avg_total_tokens"] for data in summary.values()), default=1.0) or 1.0
    max_latency = max((data["avg_total_latency_ms"] for data in summary.values()), default=1.0) or 1.0
    for data in summary.values():
        token_score = clamp01(1.0 - (data["avg_total_tokens"] / max_tokens))
        judge_score = data["avg_judge_score"] / 5.0
        if data.get("avg_judge_grounding"):
            judge_score = (0.70 * judge_score) + (0.30 * (data["avg_judge_grounding"] / 5.0))
        accuracy_score = clamp01(max(data["avg_bertscore_rescaled_f1"], data["avg_bertscore_raw_f1"], judge_score))
        latency_score = clamp01(1.0 - (data["avg_total_latency_ms"] / max_latency))
        has_retrieval = data.get("avg_retrieved_chunk_count", 0.0) > 0
        citation_score = data["avg_citation_correctness"] if has_retrieval else 0.0
        duplicate_score = (1.0 - data["avg_duplicate_chunk_ratio"]) if has_retrieval else 0.0
        engineering_score = clamp01(
            (
                data["retrieval_hit_rate"]
                + citation_score
                + (1.0 - data["avg_fabricated_citation_rate"])
                + duplicate_score
            )
            / 4.0
        )
        data["hackathon_token_reduction_score"] = token_score * 100.0
        data["hackathon_answer_accuracy_score"] = accuracy_score * 100.0
        data["hackathon_performance_latency_score"] = latency_score * 100.0
        data["hackathon_engineering_storytelling_score"] = engineering_score * 100.0
        data["hackathon_weighted_score"] = 100.0 * (
            PIPELINE_WEIGHTS["token_reduction"] * token_score
            + PIPELINE_WEIGHTS["answer_accuracy"] * accuracy_score
            + PIPELINE_WEIGHTS["performance_latency"] * latency_score
            + PIPELINE_WEIGHTS["engineering_storytelling"] * engineering_score
        )


def duplicate_ratio(texts: list[str]) -> float:
    normalized = [normalize_text(text) for text in texts if text]
    if not normalized:
        return 0.0
    counts = Counter(normalized)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    return duplicates / len(normalized)


def useful_chunk_ratio(question: str, sources: list[dict[str, Any]]) -> float:
    if not sources:
        return 0.0
    useful = 0
    for source in sources:
        if lexical_overlap(question, str(source.get("snippet") or "")) > 0:
            useful += 1
    return useful / len(sources)


def citation_correctness(answer: str, sources: list[dict[str, Any]]) -> float:
    citations = extract_citations(answer)
    if not citations:
        return 0.0 if sources else 1.0
    available = {str(index) for index in range(1, len(sources) + 1)}
    correct = sum(1 for citation in citations if citation in available)
    return correct / len(citations)


def extract_citations(text: str) -> list[str]:
    return re.findall(r"\[(\d+)\]", text)


def lexical_overlap(left: str, right: str) -> float:
    left_terms = content_terms(left)
    right_terms = content_terms(right)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms)


def unsupported_claim_estimate(answer: str, context: str) -> float:
    sentences = [sentence.strip() for sentence in re.split(r"[.!?]+", answer) if sentence.strip()]
    if not sentences:
        return 0.0
    unsupported = sum(1 for sentence in sentences if lexical_overlap(sentence, context) < 0.15)
    return unsupported / len(sentences)


def content_terms(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
    }
    return {term for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower()) if term not in stopwords}


def normalize_source_set(sources: list[Any]) -> set[str]:
    return {normalize_text(str(source)) for source in sources if str(source).strip()}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
