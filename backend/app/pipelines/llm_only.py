from __future__ import annotations

import time

from app.core.config import Settings
from app.models.api import AskRequest, RetrievalInfo
from app.pipelines.base import BasePipeline
from app.services.llm import LLMClient


class LLMOnlyPipeline(BasePipeline):
    pipeline_name = "llm-only"

    def __init__(self, settings: Settings, llm_client: LLMClient):
        self.settings = settings
        self.llm_client = llm_client

    async def run(self, payload: AskRequest):
        started = time.perf_counter()
        prompt = (
            "Answer the following scientific research question as accurately as possible.\n"
            "If uncertain, say what is uncertain.\n\n"
            f"Question: {payload.question}"
        )
        answer, prompt_tokens, completion_tokens = await self.llm_client.complete(prompt)
        latency_ms = (time.perf_counter() - started) * 1000
        cost = self._estimate_cost(
            prompt_tokens,
            completion_tokens,
            self.settings.providers.llm.pricing.prompt_per_1k,
            self.settings.providers.llm.pricing.completion_per_1k,
        )
        return self._build_response(
            answer=answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            sources=[],
            retrieval_info=RetrievalInfo(mode="none"),
            retrieval_ms=0.0,
            generation_ms=latency_ms,
            estimated_cost=cost,
        )
