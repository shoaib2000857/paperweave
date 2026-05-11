from fastapi import APIRouter, Request

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
    llm_only = await container.llm_only_pipeline.run(payload)
    basic_rag = await container.basic_rag_pipeline.run(payload)
    graphrag = await container.graphrag_pipeline.run(payload)
    return AskAllResponse(
        question=payload.question,
        llm_only=llm_only,
        basic_rag=basic_rag,
        graphrag=graphrag,
    )
