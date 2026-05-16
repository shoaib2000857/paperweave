from __future__ import annotations

import logging
from functools import lru_cache
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)
_BERTSCORE_LOCK = Lock()


@lru_cache(maxsize=1)
def _get_hf_bertscore_metric() -> Any:
    import evaluate

    return evaluate.load("bertscore")


@lru_cache(maxsize=4)
def _get_local_scorer(model_type: str, lang: str, device: str) -> Any:
    from bert_score import BERTScorer

    scorer = BERTScorer(
        model_type=model_type,
        lang=lang,
        rescale_with_baseline=False,
        device=device,
    )
    # Some tokenizers expose absurdly large max lengths that break downstream Rust usize conversions.
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
    backend: str = "evaluate",
) -> list[dict[str, float]]:
    if len(candidates) != len(references):
        raise ValueError("BERTScore candidates and references must have the same length")
    if not candidates:
        return []

    backend = backend.lower()
    if backend == "evaluate":
        return _score_with_hf_evaluate(
            candidates,
            references,
            model_type=model_type,
            batch_size=batch_size,
            rescale_with_baseline=rescale_with_baseline,
            lang=lang,
            device=device,
        )
    return _score_with_local_scorer(
        candidates,
        references,
        model_type=model_type,
        batch_size=batch_size,
        rescale_with_baseline=rescale_with_baseline,
        lang=lang,
        device=device,
    )


def _score_with_hf_evaluate(
    candidates: list[str],
    references: list[str],
    *,
    model_type: str,
    batch_size: int,
    rescale_with_baseline: bool,
    lang: str,
    device: str,
) -> list[dict[str, float]]:
    metric = _get_hf_bertscore_metric()
    try:
        with _BERTSCORE_LOCK:
            raw = metric.compute(
                predictions=candidates,
                references=references,
                model_type=model_type,
                batch_size=batch_size,
                lang=lang,
                device=device,
                rescale_with_baseline=False,
            )
            if rescale_with_baseline:
                rescaled = metric.compute(
                    predictions=candidates,
                    references=references,
                    model_type=model_type,
                    batch_size=batch_size,
                    lang=lang,
                    device=device,
                    rescale_with_baseline=True,
                )
            else:
                rescaled = raw
    except OverflowError as exc:
        logger.warning(
            "BERTScore evaluate backend hit tokenizer overflow for %s on %s; falling back to local scorer: %s",
            model_type,
            device,
            exc,
        )
        return _score_with_local_scorer(
            candidates,
            references,
            model_type=model_type,
            batch_size=batch_size,
            rescale_with_baseline=rescale_with_baseline,
            lang=lang,
            device=device,
        )

    return [
        {
            "precision": float(raw["precision"][index]),
            "recall": float(raw["recall"][index]),
            "raw_f1": float(raw["f1"][index]),
            "rescaled_precision": float(rescaled["precision"][index]),
            "rescaled_recall": float(rescaled["recall"][index]),
            "rescaled_f1": float(rescaled["f1"][index]),
        }
        for index in range(len(candidates))
    ]


def _score_with_local_scorer(
    candidates: list[str],
    references: list[str],
    *,
    model_type: str,
    batch_size: int,
    rescale_with_baseline: bool,
    lang: str,
    device: str,
) -> list[dict[str, float]]:
    scorer = _get_local_scorer(model_type, lang, device)
    with _BERTSCORE_LOCK:
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
        logger.warning(
            "BERTScore baseline unavailable for %s/%s; using raw scores for rescaled metrics: %s",
            lang,
            model_type,
            exc,
        )
        return None


def evaluate_bertscore(
    records: list[dict[str, Any]],
    model_type: str,
    batch_size: int = 8,
    rescale_with_baseline: bool = True,
    backend: str = "evaluate",
    device: str = "cpu",
) -> dict[str, Any]:
    evaluable = [record for record in records if record.get("answer") and record.get("ground_truth")]
    if not evaluable:
        return {"records": [], "summary": {}, "backend": backend}

    candidates = [record["answer"] for record in evaluable]
    references = [record["ground_truth"] for record in evaluable]
    scored_pairs = score_bertscore_pairs(
        candidates,
        references,
        model_type=model_type,
        batch_size=batch_size,
        rescale_with_baseline=rescale_with_baseline,
        device=device,
        backend=backend,
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

    return {
        "records": results,
        "summary": _summarize(results),
        "model_type": model_type,
        "backend": backend,
        "device": device,
        "rescale_with_baseline": rescale_with_baseline,
    }


def _summarize(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    pipelines = sorted({result["pipeline"] for result in results})
    summary: dict[str, dict[str, float]] = {}
    for pipeline in pipelines:
        items = [result for result in results if result["pipeline"] == pipeline]
        avg_raw_f1 = _avg([item["raw_f1"] for item in items])
        avg_rescaled_f1 = _avg([item["rescaled_f1"] for item in items])
        summary[pipeline] = {
            "avg_precision": _avg([item["precision"] for item in items]),
            "avg_recall": _avg([item["recall"] for item in items]),
            "avg_raw_f1": avg_raw_f1,
            "avg_rescaled_f1": avg_rescaled_f1,
            "raw_f1_bonus_pass": avg_raw_f1 >= 0.88,
            "rescaled_f1_bonus_pass": avg_rescaled_f1 >= 0.55,
            "bonus_pass": avg_rescaled_f1 >= 0.55,
        }
    return summary


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
