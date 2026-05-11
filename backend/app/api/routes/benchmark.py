from fastapi import APIRouter, Request

from app.models.api import BenchmarkRequest, BenchmarkResponse

router = APIRouter(tags=["benchmark"])


@router.post("/benchmark", response_model=BenchmarkResponse)
async def run_benchmark(payload: BenchmarkRequest, request: Request) -> BenchmarkResponse:
    return await request.app.state.container.benchmark_service.run(payload)
