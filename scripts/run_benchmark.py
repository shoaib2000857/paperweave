#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import Iterable
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


def log_step(message: str) -> None:
    print(f"[benchmark] {message}", flush=True)


def checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "benchmark_checkpoint.json"


def records_path(output_dir: Path) -> Path:
    return output_dir / "benchmark_pipeline_records.json"


def record_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record["question_id"]), str(record["pipeline"])


def load_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"Invalid checkpoint format in {path}")
    return records


def write_pipeline_checkpoint(
    *,
    output_dir: Path,
    dataset: str,
    pipelines: Iterable[str],
    top_k: int | None,
    question_count: int,
    records: list[dict[str, Any]],
) -> None:
    payload = {
        "dataset": dataset,
        "question_count": question_count,
        "pipelines": list(pipelines),
        "top_k": top_k,
        "records": records,
    }
    write_json(checkpoint_path(output_dir), payload)
    write_json(records_path(output_dir), payload)


async def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    log_step(f"Loading benchmark dataset from {args.dataset}")

    container = build_container()
    if args.judge_model:
        container.settings.providers.judge.model = args.judge_model
        container.judge_client = LLMClient(
            settings=container.settings,
            provider=LLMProviderFactory(container.settings).build_judge_provider(),
        )
        log_step(f"Overrode judge model to {args.judge_model}")

    questions = load_eval_dataset(args.dataset, limit=args.limit)
    log_step(f"Loaded {len(questions)} questions")
    log_step(f"Pipelines: {', '.join(args.pipelines)}")
    existing_records: list[dict[str, Any]] = []
    if args.resume or args.score_only_existing:
        checkpoint = checkpoint_path(output_dir)
        existing_records = load_checkpoint(checkpoint)
        log_step(f"Loaded {len(existing_records)} existing pipeline records from {checkpoint}")
    records = await run_pipeline_benchmark(
        questions=questions,
        container=container,
        top_k=args.top_k,
        pipelines=args.pipelines,
        output_dir=output_dir,
        dataset=str(args.dataset),
        checkpoint_every=args.checkpoint_every,
        existing_records=existing_records,
        score_only_existing=args.score_only_existing,
    )
    log_step(f"Finished pipeline execution for {len(records)} pipeline runs")
    log_step(f"Writing {records_path(output_dir)}")
    write_pipeline_checkpoint(
        output_dir=output_dir,
        dataset=str(args.dataset),
        pipelines=args.pipelines,
        top_k=args.top_k,
        question_count=len(questions),
        records=records,
    )

    log_step("Computing token metrics")
    add_token_metrics(records)
    log_step("Computing retrieval metrics")
    add_retrieval_metrics(records)
    log_step("Computing hallucination metrics")
    add_hallucination_metrics(records)

    bertscore_results: dict[str, Any] | None = None
    if not args.skip_bertscore:
        log_step(
            "Starting BERTScore "
            f"(backend={container.settings.evaluation.bertscore_backend}, "
            f"model={args.bertscore_model or container.settings.evaluation.bertscore_model}, "
            f"device={container.settings.evaluation.bertscore_device})"
        )
        bertscore_results = evaluate_bertscore(
            records,
            model_type=args.bertscore_model or container.settings.evaluation.bertscore_model,
            batch_size=args.bertscore_batch_size,
            rescale_with_baseline=not args.no_bertscore_rescale,
            device=container.settings.evaluation.bertscore_device,
            backend=container.settings.evaluation.bertscore_backend,
        )
        log_step("Finished BERTScore")
        log_step(f"Writing {output_dir / 'bertscore_results.json'}")
        write_json(output_dir / "bertscore_results.json", bertscore_results)

    judge_results: dict[str, Any] | None = None
    if args.judge:
        log_step(
            "Starting judge evaluation "
            f"(provider={container.settings.providers.judge.provider}, "
            f"model={container.settings.providers.judge.model})"
        )
        judge_results = await evaluate_with_judge(records, container.judge_client)
        log_step("Finished judge evaluation")
        log_step(f"Writing {output_dir / 'judge_results.json'}")
        write_json(output_dir / "judge_results.json", judge_results)

    log_step("Summarizing pipeline records")
    summary = summarize_pipeline_records(records)
    payload = {
        "dataset": str(args.dataset),
        "question_count": len(questions),
        "pipelines": list(args.pipelines),
        "top_k": args.top_k,
        "records": records,
        "summary": summary,
    }
    log_step(f"Writing {output_dir / 'benchmark_results.json'}")
    write_json(output_dir / "benchmark_results.json", payload)
    log_step(f"Writing {output_dir / 'leaderboard.json'}")
    write_json(output_dir / "leaderboard.json", summary)
    log_step(f"Writing CSV exports under {output_dir}")
    write_csv_exports(output_dir, records, summary)
    log_step(f"Generating charts under {report_dir}")
    generate_visualizations(report_dir, summary)
    log_step(f"Writing markdown report to {report_dir / 'summary_report.md'}")
    generate_markdown_report(
        report_dir / "summary_report.md",
        records=records,
        summary=summary,
        bertscore_results=bertscore_results,
        judge_results=judge_results,
    )
    log_step("Writing final report to evaluation/final_report.md")
    generate_markdown_report(
        "evaluation/final_report.md",
        records=records,
        summary=summary,
        bertscore_results=bertscore_results,
        judge_results=judge_results,
    )
    log_step(f"Benchmark complete. Results written to {output_dir}")
    log_step("Final report written to evaluation/final_report.md")


async def run_pipeline_benchmark(
    questions: list[EvalQuestion],
    container: Any,
    top_k: int | None,
    pipelines: tuple[str, ...],
    output_dir: Path,
    dataset: str,
    checkpoint_every: int,
    existing_records: list[dict[str, Any]] | None = None,
    score_only_existing: bool = False,
) -> list[dict[str, Any]]:
    pipeline_map = {
        "llm-only": container.llm_only_pipeline,
        "basic-rag": container.basic_rag_pipeline,
        "graphrag": container.graphrag_pipeline,
    }
    records: list[dict[str, Any]] = list(existing_records or [])
    completed_map = {record_key(record): record for record in records}
    total_questions = len(questions)
    total_runs = total_questions * len(pipelines)
    completed_runs = len(completed_map)
    if score_only_existing:
        missing = [
            (question.id, pipeline_name)
            for question in questions
            for pipeline_name in pipelines
            if (question.id, pipeline_name) not in completed_map
        ]
        if missing:
            raise RuntimeError(
                f"--score-only-existing requested, but {len(missing)} pipeline runs are missing from the checkpoint"
            )
        log_step("Score-only mode enabled; skipping pipeline generation and reusing checkpointed answers")
        return records
    for question_index, question in enumerate(questions, start=1):
        log_step(f"Question {question_index}/{total_questions}: {question.id} | {question.question}")
        for pipeline_index, pipeline_name in enumerate(pipelines, start=1):
            key = (question.id, pipeline_name)
            if key in completed_map:
                log_step(
                    f"  Skipping existing pipeline {pipeline_index}/{len(pipelines)} for question "
                    f"{question_index}/{total_questions}: {pipeline_name} | progress={completed_runs}/{total_runs}"
                )
                continue
            pipeline = pipeline_map[pipeline_name]
            request = AskRequest(question=question.question, top_k=top_k, reference_answer=question.ground_truth)
            started = time.perf_counter()
            log_step(
                f"  Running pipeline {pipeline_index}/{len(pipelines)} for question {question_index}/{total_questions}: {pipeline_name}"
            )
            try:
                response = await pipeline.run(request)
                record = response_to_record(question, response)
                records.append(record)
                completed_map[key] = record
                completed_runs += 1
                log_step(
                    "  Completed "
                    f"{pipeline_name} | latency={record['total_latency_ms']:.0f} ms | "
                    f"tokens={record['total_tokens']} | sources={record['retrieved_chunk_count']} | "
                    f"progress={completed_runs}/{total_runs}"
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - started) * 1000
                record = failed_record(question, pipeline_name, str(exc), latency_ms)
                records.append(record)
                completed_map[key] = record
                completed_runs += 1
                log_step(
                    "  FAILED "
                    f"{pipeline_name} | latency={latency_ms:.0f} ms | error={exc} | "
                    f"progress={completed_runs}/{total_runs}"
                )
            if checkpoint_every > 0 and completed_runs % checkpoint_every == 0:
                log_step(
                    f"  Writing checkpoint at progress {completed_runs}/{total_runs} to {checkpoint_path(output_dir)}"
                )
                write_pipeline_checkpoint(
                    output_dir=output_dir,
                    dataset=dataset,
                    pipelines=pipelines,
                    top_k=top_k,
                    question_count=total_questions,
                    records=records,
                )
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
        "--checkpoint-every",
        type=int,
        default=5,
        help="Write pipeline checkpoint after every N completed pipeline runs. Use 0 to disable periodic checkpoints.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume generation from evaluation/outputs/benchmark_checkpoint.json if it exists.",
    )
    parser.add_argument(
        "--score-only-existing",
        action="store_true",
        help="Skip pipeline generation and only score existing checkpointed records.",
    )
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
