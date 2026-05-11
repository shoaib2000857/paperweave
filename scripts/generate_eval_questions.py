#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from app.core.config import get_settings


QUESTIONS = [
    {
        "id": "q001",
        "category": "lookup",
        "question": "What is retrieval augmented generation?",
        "reference_answer": "Retrieval augmented generation combines external retrieval with an LLM so the model answers using fetched supporting context rather than relying only on parametric memory.",
    },
    {
        "id": "q002",
        "category": "lookup",
        "question": "What is BERTScore used for in answer evaluation?",
        "reference_answer": "BERTScore measures semantic similarity between a generated answer and a reference answer using contextual embeddings.",
    },
    {
        "id": "q003",
        "category": "multi_hop_reasoning",
        "question": "How did retrieval-augmented methods evolve into graph-based RAG for scientific question answering?",
        "reference_answer": "A strong answer should connect dense or chunk retrieval, entity extraction, relationship modeling, and multi-hop reasoning over linked concepts or documents.",
    },
    {
        "id": "q004",
        "category": "comparison",
        "question": "Compare Basic RAG and GraphRAG for multi-hop reasoning over scientific papers.",
        "reference_answer": "Basic RAG retrieves semantically similar chunks, while GraphRAG additionally reasons over entities and relationships, which can reduce context size and improve multi-hop evidence assembly.",
    },
    {
        "id": "q005",
        "category": "graph_traversal_reasoning",
        "question": "Which papers, datasets, and methods connect hallucination detection with retrieval augmentation?",
        "reference_answer": "A strong answer should identify linked papers, the datasets they evaluate on, and the methods or retrieval strategies that relate hallucination detection to retrieval augmentation.",
    },
]


def main() -> None:
    settings = get_settings()
    output = Path(settings.paths.eval_questions_dir) / "benchmark_questions.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(QUESTIONS, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
