#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
for path in (REPO_ROOT, BACKEND_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.core.dependencies import build_container
from app.models.api import AskRequest
from app.services.llm import LLMClient
from app.services.providers import LLMProviderFactory
from evaluation.bertscore_eval import evaluate_bertscore
from evaluation.dataset import EvalQuestion, load_eval_dataset
from evaluation.judge import evaluate_with_judge
from evaluation.metrics import (
    add_hallucination_metrics,
    add_retrieval_metrics,
    add_token_metrics,
    failed_record,
    response_to_record,
    summarize_pipeline_records,
)
from evaluation.reporting import generate_markdown_report, generate_visualizations, write_csv_exports, write_json


PIPELINES = ("llm-only", "basic-rag", "graphrag")


async def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    container = build_container()
    if args.judge_model:
        container.settings.providers.judge.model = args.judge_model
        container.judge_client = LLMClient(
            settings=container.settings,
            provider=LLMProviderFactory(container.settings).build_judge_provider(),
        )

    questions = load_eval_dataset(args.dataset, limit=args.limit)
    records = await run_pipeline_benchmark(
        questions=questions,
        container=container,
        top_k=args.top_k,
        pipelines=args.pipelines,
    )

    add_token_metrics(records)
    add_retrieval_metrics(records)
    add_hallucination_metrics(records)

    bertscore_results: dict[str, Any] | None = None
    if not args.skip_bertscore:
        bertscore_results = evaluate_bertscore(
            records,
            model_type=args.bertscore_model or container.settings.evaluation.bertscore_model,
            batch_size=args.bertscore_batch_size,
            rescale_with_baseline=not args.no_bertscore_rescale,
        )
        write_json(output_dir / "bertscore_results.json", bertscore_results)

    judge_results: dict[str, Any] | None = None
    if args.judge:
        judge_results = await evaluate_with_judge(records, container.judge_client)
        write_json(output_dir / "judge_results.json", judge_results)

    summary = summarize_pipeline_records(records)
    payload = {
        "dataset": str(args.dataset),
        "question_count": len(questions),
        "pipelines": list(args.pipelines),
        "top_k": args.top_k,
        "records": records,
        "summary": summary,
    }
    write_json(output_dir / "benchmark_results.json", payload)
    write_json(output_dir / "leaderboard.json", summary)
    write_csv_exports(output_dir, records, summary)
    generate_visualizations(report_dir, summary)
    generate_markdown_report(
        report_dir / "summary_report.md",
        records=records,
        summary=summary,
        bertscore_results=bertscore_results,
        judge_results=judge_results,
    )
    generate_markdown_report(
        "evaluation/final_report.md",
        records=records,
        summary=summary,
        bertscore_results=bertscore_results,
        judge_results=judge_results,
    )
    print(f"Benchmark complete. Results written to {output_dir}")
    print(f"Final report written to evaluation/final_report.md")


async def run_pipeline_benchmark(
    questions: list[EvalQuestion],
    container: Any,
    top_k: int | None,
    pipelines: tuple[str, ...],
) -> list[dict[str, Any]]:
    pipeline_map = {
        "llm-only": container.llm_only_pipeline,
        "basic-rag": container.basic_rag_pipeline,
        "graphrag": container.graphrag_pipeline,
    }
    records: list[dict[str, Any]] = []
    for question in questions:
        for pipeline_name in pipelines:
            pipeline = pipeline_map[pipeline_name]
            request = AskRequest(question=question.question, top_k=top_k, reference_answer=question.ground_truth)
            started = time.perf_counter()
            try:
                response = await pipeline.run(request)
                records.append(response_to_record(question, response))
            except Exception as exc:
                latency_ms = (time.perf_counter() - started) * 1000
                records.append(failed_record(question, pipeline_name, str(exc), latency_ms))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PaperWeave unified evaluation benchmark.")
    parser.add_argument(
        "--dataset",
        default="evaluation/datasets/sample_scientific_qa.json",
        help="Evaluation dataset JSON path.",
    )
    parser.add_argument("--output-dir", default="evaluation/outputs", help="Directory for JSON and CSV outputs.")
    parser.add_argument("--report-dir", default="evaluation/reports", help="Directory for markdown and chart outputs.")
    parser.add_argument("--judge-model", default=None, help="Override configured judge model name.")
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval k for RAG pipelines.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions.")
    parser.add_argument(
        "--pipelines",
        nargs="+",
        choices=PIPELINES,
        default=PIPELINES,
        help="Pipelines to benchmark.",
    )
    parser.add_argument("--judge", action="store_true", help="Enable LLM-as-a-judge evaluation.")
    parser.add_argument("--skip-bertscore", action="store_true", help="Skip BERTScore evaluation.")
    parser.add_argument("--bertscore-model", default=None, help="Override BERTScore model.")
    parser.add_argument("--bertscore-batch-size", type=int, default=8, help="BERTScore batch size.")
    parser.add_argument("--no-bertscore-rescale", action="store_true", help="Disable BERTScore baseline rescaling.")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
