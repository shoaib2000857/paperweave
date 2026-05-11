from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes.ask import router as ask_router
from app.api.routes.benchmark import router as benchmark_router
from app.api.routes.metrics import router as metrics_router
from app.core.dependencies import build_container
from app.core.logging import configure_logging
from app.services.providers import ProviderError


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    app.state.container = build_container()
    yield


app = FastAPI(
    title="PaperWeave API",
    version="0.1.0",
    lifespan=lifespan,
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "PAPERWEAVE_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ProviderError)
async def provider_error_handler(_: Request, exc: ProviderError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": exc.as_detail()})


app.include_router(ask_router)
app.include_router(benchmark_router)
app.include_router(metrics_router)
