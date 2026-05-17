from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

_CHART_FILENAMES = (
    "token_usage.png",
    "latency.png",
    "bertscore.png",
    "judge_pass_rate.png",
    "leaderboard.png",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _output_dir() -> Path:
    for candidate in (
        _repo_root() / "evaluation" / "outputs" / "hackathon_favor10",
        _repo_root() / "evaluation" / "outputs" / "smoke",
        _repo_root() / "evaluation" / "outputs",
        _repo_root() / "benchmarks",
    ):
        if candidate.exists():
            return candidate
    raise HTTPException(status_code=404, detail="No evaluation outputs found")


def _report_dir() -> Path:
    for candidate in (
        _repo_root() / "evaluation" / "reports" / "hackathon_favor10",
        _repo_root() / "evaluation" / "reports" / "smoke",
        _repo_root() / "evaluation" / "reports",
    ):
        if candidate.exists():
            return candidate
    raise HTTPException(status_code=404, detail="No evaluation reports found")


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _load_benchmark_payload() -> dict[str, Any]:
    output_dir = _output_dir()
    benchmark_path = output_dir / "benchmark_results.json"
    payload = _read_json(benchmark_path)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Benchmark results not found at {benchmark_path}")
    return payload


def _load_summary() -> dict[str, Any]:
    payload = _load_benchmark_payload()
    summary = payload.get("summary") or {}
    leaderboard_path = _output_dir() / "leaderboard.json"
    leaderboard = _read_json(leaderboard_path)
    if isinstance(leaderboard, dict):
        summary = leaderboard
    return summary


def _artifact_payload(name: str) -> dict[str, Any]:
    output_dir = _output_dir()
    path = output_dir / name
    data = _read_json(path)
    return {
        "available": data is not None,
        "path": str(path),
        "data": data,
    }


def _leaderboard_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [{"pipeline": pipeline, **values} for pipeline, values in summary.items()]
    rows.sort(key=lambda row: row.get("hackathon_weighted_score", 0.0), reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def _latest_live_event(request: Request) -> dict[str, Any] | None:
    events = request.app.state.container.metrics_service.read_metrics().get("events", [])
    for event in reversed(events):
        if event.get("type") == "live_query_evaluation":
            return event
    return None


def _report_payload() -> dict[str, Any]:
    report_dir = _report_dir()
    markdown_path = report_dir / "summary_report.md"
    markdown = _read_text(markdown_path)
    charts = []
    for filename in _CHART_FILENAMES:
        path = report_dir / filename
        if path.exists():
            charts.append(
                {
                    "name": filename.replace(".png", "").replace("_", " ").title(),
                    "path": str(path),
                    "url": f"/evaluation/charts/{filename}",
                }
            )
    return {
        "available": markdown is not None,
        "path": str(markdown_path),
        "markdown": markdown,
        "charts": charts,
    }


@router.get("/results")
async def get_results(request: Request) -> dict[str, Any]:
    try:
        offline_benchmark = _load_benchmark_payload()
        offline_summary = offline_benchmark.get("summary") or {}
        offline_leaderboard = _leaderboard_rows(offline_summary)
    except HTTPException:
        offline_benchmark = {
            "dataset": "offline-unavailable",
            "question_count": 0,
            "pipelines": [],
            "top_k": None,
            "records": [],
            "summary": {},
        }
        offline_leaderboard = []
    latest_live = _latest_live_event(request)
    live_payload = (latest_live or {}).get("payload") if latest_live else None

    benchmark = live_payload.get("benchmark") if isinstance(live_payload, dict) else offline_benchmark
    leaderboard = live_payload.get("leaderboard") if isinstance(live_payload, dict) else offline_leaderboard

    return {
        "benchmark": benchmark,
        "leaderboard": leaderboard,
        "live": live_payload,
        "offline": {
            "benchmark": offline_benchmark,
            "leaderboard": offline_leaderboard,
        },
        "bertscore": _artifact_payload("bertscore_results.json"),
        "judge": _artifact_payload("judge_results.json"),
        "report": _report_payload(),
    }


@router.get("/live")
async def get_live_results(request: Request) -> dict[str, Any]:
    latest_live = _latest_live_event(request)
    if not latest_live:
        raise HTTPException(status_code=404, detail="No live evaluation found yet")
    return {
        "timestamp": latest_live.get("timestamp"),
        "question": latest_live.get("question"),
        "payload": latest_live.get("payload"),
    }


@router.get("/leaderboard")
async def get_leaderboard(request: Request) -> dict[str, Any]:
    latest_live = _latest_live_event(request)
    live_payload = (latest_live or {}).get("payload") if latest_live else None
    if isinstance(live_payload, dict) and isinstance(live_payload.get("leaderboard"), list):
        return {"rows": live_payload.get("leaderboard"), "source": "live"}
    summary = _load_summary()
    return {"rows": _leaderboard_rows(summary), "source": "offline"}


@router.get("/benchmark")
async def get_benchmark() -> dict[str, Any]:
    return _load_benchmark_payload()


@router.get("/bertscore")
async def get_bertscore() -> dict[str, Any]:
    return _artifact_payload("bertscore_results.json")


@router.get("/judge")
async def get_judge() -> dict[str, Any]:
    return _artifact_payload("judge_results.json")


@router.get("/report")
async def get_report() -> dict[str, Any]:
    return _report_payload()


@router.get("/charts/{chart_name}")
async def get_chart(chart_name: str) -> FileResponse:
    report_dir = _report_dir()
    path = report_dir / chart_name
    if not path.exists() or path.suffix.lower() != ".png":
        raise HTTPException(status_code=404, detail="Chart not found")
    return FileResponse(path)
