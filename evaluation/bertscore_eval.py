from __future__ import annotations

import logging
from functools import lru_cache
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)
_SCORER_LOCK = Lock()
_MISSING_BASELINE_WARNED: set[tuple[str, str]] = set()


@lru_cache(maxsize=4)
def _get_scorer(model_type: str, lang: str, device: str):
    from bert_score import BERTScorer

    scorer = BERTScorer(
        model_type=model_type,
        lang=lang,
        rescale_with_baseline=False,
        device=device,
    )
    # Some models set model_max_length to a huge value that overflows Rust's usize in tokenizers.
    scorer._tokenizer.model_max_length = 512
    return scorer


def score_bertscore_pairs(
    candidates: list[str],
    references: list[str],
    *,
    model_type: str,
    batch_size: int = 8,
    rescale_with_baseline: bool = True,
    lang: str = "en",
    device: str = "cpu",
) -> list[dict[str, float]]:
    if len(candidates) != len(references):
        raise ValueError("BERTScore candidates and references must have the same length")
    if not candidates:
        return []

    scorer = _get_scorer(model_type, lang, device)
    with _SCORER_LOCK:
        precision, recall, f1 = scorer.score(candidates, references, batch_size=batch_size)
        if rescale_with_baseline:
            baseline = _baseline_values(scorer, model_type, lang)
            if baseline is not None:
                rescaled_precision = (precision - baseline[0]) / (1 - baseline[0])
                rescaled_recall = (recall - baseline[1]) / (1 - baseline[1])
                rescaled_f1 = (f1 - baseline[2]) / (1 - baseline[2])
            else:
                rescaled_precision, rescaled_recall, rescaled_f1 = precision, recall, f1
        else:
            rescaled_precision, rescaled_recall, rescaled_f1 = precision, recall, f1

    return [
        {
            "precision": float(precision[index].item()),
            "recall": float(recall[index].item()),
            "raw_f1": float(f1[index].item()),
            "rescaled_precision": float(rescaled_precision[index].item()),
            "rescaled_recall": float(rescaled_recall[index].item()),
            "rescaled_f1": float(rescaled_f1[index].item()),
        }
        for index in range(len(candidates))
    ]


def _baseline_values(scorer: Any, model_type: str, lang: str) -> Any | None:
    try:
        return scorer.baseline_vals
    except ValueError as exc:
        # Custom/scientific models often do not ship a BERTScore baseline file.
        # Keep evaluation usable and make rescaled metrics equal raw metrics.
        key = (model_type, lang)
        if key not in _MISSING_BASELINE_WARNED:
            _MISSING_BASELINE_WARNED.add(key)
            logger.warning("BERTScore baseline unavailable for %s/%s; using raw scores for rescaled metrics: %s", lang, model_type, exc)
        return None


def evaluate_bertscore(
    records: list[dict[str, Any]],
    model_type: str,
    batch_size: int = 8,
    rescale_with_baseline: bool = True,
) -> dict[str, Any]:
    evaluable = [record for record in records if record.get("answer") and record.get("ground_truth")]
    if not evaluable:
        return {"records": [], "summary": {}}

    candidates = [record["answer"] for record in evaluable]
    references = [record["ground_truth"] for record in evaluable]
    scored_pairs = score_bertscore_pairs(
        candidates,
        references,
        model_type=model_type,
        batch_size=batch_size,
        rescale_with_baseline=rescale_with_baseline,
    )

    results: list[dict[str, Any]] = []
    for index, record in enumerate(evaluable):
        scored = scored_pairs[index]
        metrics = {
            "question_id": record["question_id"],
            "pipeline": record["pipeline"],
            "precision": scored["precision"],
            "recall": scored["recall"],
            "raw_f1": scored["raw_f1"],
            "rescaled_precision": scored["rescaled_precision"],
            "rescaled_recall": scored["rescaled_recall"],
            "rescaled_f1": scored["rescaled_f1"],
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
