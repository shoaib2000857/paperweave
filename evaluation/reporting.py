from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv_exports(output_dir: str | Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(base / "benchmark_records.csv", index=False)
    leaderboard = leaderboard_rows(summary)
    pd.DataFrame(leaderboard).to_csv(base / "leaderboard.csv", index=False)


def generate_markdown_report(
    output_path: str | Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    bertscore_results: dict[str, Any] | None = None,
    judge_results: dict[str, Any] | None = None,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PaperWeave Evaluation Report",
        "",
        f"Questions evaluated: {len({record['question_id'] for record in records})}",
        f"Pipeline runs: {len(records)}",
        "",
        "## Leaderboard",
        "",
        _markdown_table(
            leaderboard_rows(summary),
            [
                "rank",
                "pipeline",
                "hackathon_weighted_score",
                "avg_token_reduction_pct_vs_llm_only",
                "avg_total_latency_ms",
                "avg_bertscore_rescaled_f1",
                "judge_pass_rate",
            ],
        ),
        "",
        "## Hackathon Criteria",
        "",
        "- Token Reduction: 30% of weighted score, based on average total-token reduction relative to LLM-only.",
        "- Answer Accuracy: 30%, using the strongest available signal among BERTScore and judge score.",
        "- Performance / Latency: 20%, based on relative total latency.",
        "- Engineering & Storytelling: 20%, based on retrieval hit rate, citation correctness, duplicate control, and fabricated citation avoidance.",
        "",
        "## Pipeline Summary",
        "",
        _markdown_table(summary_rows(summary), ["pipeline", "count", "failures", "avg_total_tokens", "p50_total_latency_ms", "p95_total_latency_ms"]),
        "",
        "## Bonus Checks",
        "",
    ]
    if bertscore_results:
        lines.extend(_bonus_lines("BERTScore", bertscore_results.get("summary", {})))
    if judge_results:
        lines.extend(_bonus_lines("Judge", judge_results.get("summary", {})))
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- Raw benchmark records: `evaluation/outputs/benchmark_results.json`",
            "- BERTScore results: `evaluation/outputs/bertscore_results.json`",
            "- Judge results: `evaluation/outputs/judge_results.json`",
            "- Leaderboard CSV: `evaluation/outputs/leaderboard.csv`",
            "- Visualizations: `evaluation/reports/*.png`",
            "",
            "## Known Limitations",
            "",
            "- GraphRAG metrics depend on the running TigerGraph service exposing source snippets.",
            "- Token counts are estimated with the local tokenizer when upstream providers do not return usage.",
            "- Heuristic hallucination and retrieval metrics are useful diagnostics, not replacements for expert review.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_visualizations(report_dir: str | Path, summary: dict[str, Any]) -> list[str]:
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    rows = summary_rows(summary)
    if not rows:
        return []

    paths: list[str] = []
    charts = [
        ("token_usage.png", "avg_total_tokens", "Average Total Tokens", "tokens"),
        ("latency.png", "avg_total_latency_ms", "Average Total Latency", "milliseconds"),
        ("bertscore.png", "avg_bertscore_rescaled_f1", "BERTScore Rescaled F1", "F1"),
        ("judge_pass_rate.png", "judge_pass_rate", "Judge Pass Rate", "rate"),
        ("leaderboard.png", "hackathon_weighted_score", "Hackathon Weighted Score", "score"),
    ]
    for filename, key, title, ylabel in charts:
        pipelines = [row["pipeline"] for row in rows]
        values = [float(row.get(key) or 0.0) for row in rows]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(pipelines, values, color=["#2f6f73", "#c97f35", "#6c63a6"][: len(pipelines)])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("pipeline")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        chart_path = output_dir / filename
        fig.savefig(chart_path, dpi=160)
        plt.close(fig)
        paths.append(str(chart_path))
    return paths


def leaderboard_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary_rows(summary)
    rows.sort(key=lambda row: row.get("hackathon_weighted_score", 0.0), reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"pipeline": pipeline, **values} for pipeline, values in summary.items()]


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No data available._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(_format_cell(row.get(column)) for column in columns) + " |")
    return "\n".join([header, divider, *body])


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _bonus_lines(label: str, summary: dict[str, Any]) -> list[str]:
    lines = [f"### {label}", ""]
    for pipeline, values in summary.items():
        if "bonus_pass" in values:
            lines.append(f"- {pipeline}: bonus pass = {values['bonus_pass']}")
        elif "raw_f1_bonus_pass" in values:
            lines.append(
                f"- {pipeline}: raw F1 pass = {values['raw_f1_bonus_pass']}, "
                f"rescaled F1 pass = {values['rescaled_f1_bonus_pass']}"
            )
    return lines
