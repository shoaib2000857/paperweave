from __future__ import annotations

import json
import re
from typing import Any

from app.services.llm import LLMClient


async def evaluate_with_judge(records: list[dict[str, Any]], judge_client: LLMClient) -> dict[str, Any]:
    results = []
    for record in records:
        if not record.get("answer"):
            result = _empty_result(record, "No candidate answer was produced.")
        else:
            result = await _judge_one(record, judge_client)
        record["judge_score"] = result["score"]
        record["judge_pass"] = result["pass"]
        record["judge_reasoning"] = result["reasoning"]
        record["judge_factual_correctness"] = result.get("factual_correctness")
        record["judge_grounding"] = result.get("grounding")
        record["judge_completeness"] = result.get("completeness")
        record["judge_hallucination_level"] = result.get("hallucination_level")
        record["judge_scientific_accuracy"] = result.get("scientific_accuracy")
        results.append(result)
    return {"records": results, "summary": _summarize(results)}


async def _judge_one(record: dict[str, Any], judge_client: LLMClient) -> dict[str, Any]:
    prompt = (
        "You are an expert scientific QA evaluator. Grade the candidate answer against the reference answer "
        "and any retrieved context. Use strict scientific correctness standards.\n\n"
        "Scoring rubric: 1=poor, 2=weak, 3=partially correct, 4=mostly correct, 5=excellent.\n"
        "Pass only if score >= 4, hallucination_level <= 2, and the answer is scientifically accurate.\n\n"
        f"Question: {record.get('question')}\n\n"
        f"Reference answer: {record.get('ground_truth')}\n\n"
        f"Retrieved context: {record.get('retrieved_context')[:6000]}\n\n"
        f"Candidate answer: {record.get('answer')}\n\n"
        "Return only valid JSON with this schema:\n"
        "{\n"
        '  "score": 1-5,\n'
        '  "pass": true/false,\n'
        '  "factual_correctness": 1-5,\n'
        '  "grounding": 1-5,\n'
        '  "completeness": 1-5,\n'
        '  "hallucination_level": 1-5,\n'
        '  "scientific_accuracy": 1-5,\n'
        '  "reasoning": "brief reason"\n'
        "}"
    )
    try:
        text, _, _ = await judge_client.complete(prompt)
        parsed = _parse_json(text)
        score = _as_float(parsed.get("score"), 0.0)
        hallucination_level = _as_float(parsed.get("hallucination_level"), 5.0)
        return {
            "question_id": record["question_id"],
            "pipeline": record["pipeline"],
            "score": score,
            "pass": bool(parsed.get("pass")) and score >= 4.0 and hallucination_level <= 2.0,
            "factual_correctness": _as_float(parsed.get("factual_correctness"), score),
            "grounding": _as_float(parsed.get("grounding"), 0.0),
            "completeness": _as_float(parsed.get("completeness"), 0.0),
            "hallucination_level": hallucination_level,
            "scientific_accuracy": _as_float(parsed.get("scientific_accuracy"), score),
            "reasoning": str(parsed.get("reasoning") or text).strip(),
        }
    except Exception as exc:
        return _empty_result(record, f"Judge evaluation failed: {exc}")


def _empty_result(record: dict[str, Any], reasoning: str) -> dict[str, Any]:
    return {
        "question_id": record["question_id"],
        "pipeline": record["pipeline"],
        "score": 0.0,
        "pass": False,
        "factual_correctness": 0.0,
        "grounding": 0.0,
        "completeness": 0.0,
        "hallucination_level": 5.0,
        "scientific_accuracy": 0.0,
        "reasoning": reasoning,
    }


def _parse_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _summarize(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for pipeline in sorted({result["pipeline"] for result in results}):
        items = [result for result in results if result["pipeline"] == pipeline]
        pass_rate = sum(1 for item in items if item["pass"]) / len(items) if items else 0.0
        summary[pipeline] = {
            "pass_rate": pass_rate,
            "pass_rate_pct": pass_rate * 100.0,
            "avg_score": _avg([item["score"] for item in items]),
            "hallucination_rate": _avg([item["hallucination_level"] for item in items]),
            "bonus_pass": pass_rate >= 0.90,
        }
    return summary


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
