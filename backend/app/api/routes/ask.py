import asyncio
import time

from fastapi import APIRouter, Request
from fastapi.encoders import jsonable_encoder

from app.models.api import (
    AskAllResponse,
    AskRequest,
    AskResponse,
    RetrievalInfo,
    TimingBreakdown,
    TokenUsage,
)

router = APIRouter(tags=["ask"])


def _container(request: Request):
    return request.app.state.container


def _error_response(pipeline_name: str, message: str, latency_ms: float) -> AskResponse:
    """Minimal AskResponse placeholder used when a pipeline raises an unhandled exception."""
    return AskResponse(
        pipeline=pipeline_name,
        answer=f"[{pipeline_name}] Pipeline error: {message}",
        tokens=TokenUsage(),
        latency=latency_ms,
        estimated_cost=0.0,
        sources=[],
        retrieval_info=RetrievalInfo(
            mode="error",
            raw={"error": message},
        ),
        timing_breakdown=TimingBreakdown(total_ms=latency_ms),
        raw={"status": "error", "error": message},
    )


@router.post("/ask/llm-only", response_model=AskResponse)
async def ask_llm_only(payload: AskRequest, request: Request) -> AskResponse:
    return await _container(request).llm_only_pipeline.run(payload)


@router.post("/ask/basic-rag", response_model=AskResponse)
async def ask_basic_rag(payload: AskRequest, request: Request) -> AskResponse:
    return await _container(request).basic_rag_pipeline.run(payload)


@router.post("/ask/graphrag", response_model=AskResponse)
async def ask_graphrag(payload: AskRequest, request: Request) -> AskResponse:
    return await _container(request).graphrag_pipeline.run(payload)


@router.post("/ask/all", response_model=AskAllResponse)
async def ask_all(payload: AskRequest, request: Request) -> AskAllResponse:
    container = _container(request)

    pipeline_starts = {name: time.perf_counter() for name in ("llm-only", "basic-rag", "graphrag")}

    # Run all three pipelines concurrently; capture exceptions rather than propagating them.
    results = await asyncio.gather(
        container.llm_only_pipeline.run(payload),
        container.basic_rag_pipeline.run(payload),
        container.graphrag_pipeline.run(payload),
        return_exceptions=True,
    )

    pipeline_names = ["llm-only", "basic-rag", "graphrag"]
    errors: dict[str, str] = {}
    responses: dict[str, AskResponse] = {}

    for name, result in zip(pipeline_names, results):
        if isinstance(result, BaseException):
            elapsed_ms = (time.perf_counter() - pipeline_starts[name]) * 1000
            error_msg = str(result)
            errors[name] = error_msg
            responses[name] = _error_response(name, error_msg, elapsed_ms)
        else:
            responses[name] = result

    # Run live evaluation; if it fails, return partial results rather than crashing.
    try:
        live_eval = await container.live_evaluation_service.evaluate_live_query(
            question=payload.question,
            responses=responses,
            reference_answer=payload.reference_answer,
        )
    except Exception as exc:
        live_eval = {
            "pipelines": {},
            "leaderboard": [],
            "global_metrics": {"evaluation_error": str(exc)},
        }
        errors["evaluation"] = str(exc)

    try:
        container.metrics_service.record(
            jsonable_encoder({
                "type": "live_query_evaluation",
                "question": payload.question,
                "leaderboard": live_eval.get("leaderboard", []),
                "global_metrics": live_eval.get("global_metrics", {}),
                "pipeline_errors": errors,
            })
        )
    except Exception:
        pass  # Metrics recording is non-critical

    return AskAllResponse(
        question=payload.question,
        pipelines=live_eval.get("pipelines", {}),
        leaderboard=live_eval.get("leaderboard", []),
        global_metrics=live_eval.get("global_metrics", {}),
        llm_only=responses.get("llm-only"),
        basic_rag=responses.get("basic-rag"),
        graphrag=responses.get("graphrag"),
        errors=errors,
    )
