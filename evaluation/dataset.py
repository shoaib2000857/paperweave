from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    ground_truth: str
    category: str = "uncategorized"
    difficulty: str = "unknown"
    sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_eval_dataset(path: str | Path, limit: int | None = None) -> list[EvalQuestion]:
    dataset_path = Path(path)
    records = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Evaluation dataset must be a JSON array: {dataset_path}")

    questions = [_normalize_record(record, index) for index, record in enumerate(records, start=1)]
    if limit is not None:
        return questions[:limit]
    return questions


def _normalize_record(record: dict[str, Any], index: int) -> EvalQuestion:
    ground_truth = record.get("ground_truth") or record.get("reference_answer") or ""
    if not record.get("question"):
        raise ValueError(f"Dataset record {index} is missing required field: question")
    if not ground_truth:
        raise ValueError(f"Dataset record {index} is missing required field: ground_truth")

    sources = record.get("sources") or []
    if not isinstance(sources, list):
        sources = [str(sources)]

    known = {"id", "question", "ground_truth", "reference_answer", "category", "difficulty", "sources"}
    return EvalQuestion(
        id=str(record.get("id") or f"q{index:04d}"),
        question=str(record["question"]),
        ground_truth=str(ground_truth),
        category=str(record.get("category") or "uncategorized"),
        difficulty=str(record.get("difficulty") or "unknown"),
        sources=[str(source) for source in sources],
        metadata={key: value for key, value in record.items() if key not in known},
    )


def write_dataset_template(path: str | Path, count: int = 3) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template = [
        {
            "id": f"q{index:03d}",
            "question": "Replace with a scientific QA benchmark question.",
            "ground_truth": "Replace with the expected answer grounded in the benchmark corpus.",
            "category": "lookup | comparison | multi_hop_reasoning | summarization | graph_traversal_reasoning",
            "difficulty": "easy | medium | hard",
            "sources": ["paper-or-document-id"],
        }
        for index in range(1, count + 1)
    ]
    output_path.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
