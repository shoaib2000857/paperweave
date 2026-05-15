from __future__ import annotations

from typing import Any

def evaluate_bertscore(
    records: list[dict[str, Any]],
    model_type: str,
    batch_size: int = 8,
    rescale_with_baseline: bool = True,
) -> dict[str, Any]:
    evaluable = [record for record in records if record.get("answer") and record.get("ground_truth")]
    if not evaluable:
        return {"records": [], "summary": {}}

    from bert_score import BERTScorer

    candidates = [record["answer"] for record in evaluable]
    references = [record["ground_truth"] for record in evaluable]

    scorer = BERTScorer(model_type=model_type, batch_size=batch_size, lang="en", rescale_with_baseline=False)
    # Some models set model_max_length to a huge value that overflows Rust's usize in tokenizers.
    scorer._tokenizer.model_max_length = 512

    precision, recall, f1 = scorer.score(candidates, references)

    rescaled_scorer = BERTScorer(
        model_type=model_type, batch_size=batch_size, lang="en", rescale_with_baseline=rescale_with_baseline
    )
    rescaled_scorer._tokenizer.model_max_length = 512
    rescaled_precision, rescaled_recall, rescaled_f1 = rescaled_scorer.score(candidates, references)

    results: list[dict[str, Any]] = []
    for index, record in enumerate(evaluable):
        metrics = {
            "question_id": record["question_id"],
            "pipeline": record["pipeline"],
            "precision": float(precision[index].item()),
            "recall": float(recall[index].item()),
            "raw_f1": float(f1[index].item()),
            "rescaled_precision": float(rescaled_precision[index].item()),
            "rescaled_recall": float(rescaled_recall[index].item()),
            "rescaled_f1": float(rescaled_f1[index].item()),
        }
        record["bertscore_precision"] = metrics["precision"]
        record["bertscore_recall"] = metrics["recall"]
        record["bertscore_raw_f1"] = metrics["raw_f1"]
        record["bertscore_rescaled_precision"] = metrics["rescaled_precision"]
        record["bertscore_rescaled_recall"] = metrics["rescaled_recall"]
        record["bertscore_rescaled_f1"] = metrics["rescaled_f1"]
        results.append(metrics)

    return {"records": results, "summary": _summarize(results), "model_type": model_type}


def _summarize(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    pipelines = sorted({result["pipeline"] for result in results})
    summary: dict[str, dict[str, float]] = {}
    for pipeline in pipelines:
        items = [result for result in results if result["pipeline"] == pipeline]
        summary[pipeline] = {
            "avg_precision": _avg([item["precision"] for item in items]),
            "avg_recall": _avg([item["recall"] for item in items]),
            "avg_raw_f1": _avg([item["raw_f1"] for item in items]),
            "avg_rescaled_f1": _avg([item["rescaled_f1"] for item in items]),
            "raw_f1_bonus_pass": _avg([item["raw_f1"] for item in items]) >= 0.88,
            "rescaled_f1_bonus_pass": _avg([item["rescaled_f1"] for item in items]) >= 0.55,
        }
    return summary


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
