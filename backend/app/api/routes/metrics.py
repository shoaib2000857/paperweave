from fastapi import APIRouter, Request

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def get_metrics(request: Request) -> dict:
    return request.app.state.container.metrics_service.read_metrics()


@router.get("/health")
async def health(request: Request) -> dict:
    settings = request.app.state.container.settings
    return {
        "status": "ok",
        "service": settings.app.name,
        "environment": settings.app.environment,
    }
