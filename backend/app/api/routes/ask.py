import asyncio

from fastapi import APIRouter, Request
from fastapi.encoders import jsonable_encoder

from app.models.api import AskAllResponse, AskRequest, AskResponse

router = APIRouter(tags=["ask"])


def _container(request: Request):
    return request.app.state.container


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
    llm_only, basic_rag, graphrag = await asyncio.gather(
        container.llm_only_pipeline.run(payload),
        container.basic_rag_pipeline.run(payload),
        container.graphrag_pipeline.run(payload),
    )
    responses = {
        "llm-only": llm_only,
        "basic-rag": basic_rag,
        "graphrag": graphrag,
    }
    live_eval = await container.live_evaluation_service.evaluate_live_query(
        question=payload.question,
        responses=responses,
        reference_answer=payload.reference_answer,
    )
    container.metrics_service.record(
        jsonable_encoder({
            "type": "live_query_evaluation",
            "question": payload.question,
            "leaderboard": live_eval.get("leaderboard", []),
            "global_metrics": live_eval.get("global_metrics", {}),
            "payload": live_eval,
        })
    )
    return AskAllResponse(
        question=payload.question,
        pipelines=live_eval.get("pipelines", {}),
        leaderboard=live_eval.get("leaderboard", []),
        global_metrics=live_eval.get("global_metrics", {}),
        llm_only=llm_only,
        basic_rag=basic_rag,
        graphrag=graphrag,
    )
