from __future__ import annotations

import time

from app.core.config import Settings
from app.models.api import AskRequest, RetrievalInfo, SourceRecord
from app.pipelines.base import BasePipeline
from app.services.llm import LLMClient
from app.services.providers import EmbeddingProvider
from app.storage.dataset_store import DatasetStore
from app.storage.rag_store import BasicRAGStore


class BasicRAGPipeline(BasePipeline):
    pipeline_name = "basic-rag"

    def __init__(
        self,
        settings: Settings,
        llm_client: LLMClient,
        embedding_provider: EmbeddingProvider,
        rag_store: BasicRAGStore,
        dataset_store: DatasetStore,
    ):
        self.settings = settings
        self.llm_client = llm_client
        self.embedding_provider = embedding_provider
        self.rag_store = rag_store
        self.dataset_store = dataset_store

    async def run(self, payload: AskRequest):
        started = time.perf_counter()
        retrieval_start = time.perf_counter()
        top_k = payload.top_k or self.settings.retrieval.basic_rag_top_k
        query_embedding = self.embedding_provider.embed_query(payload.question)
        hits = self.rag_store.search(query_embedding, top_k=top_k)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        sources = [
            SourceRecord(
                id=str(hit.get("paper_id", hit.get("id", "unknown"))),
                title=hit.get("title"),
                snippet=hit["text"][:600],
                score=hit.get("score"),
                metadata={"chunk_id": hit.get("chunk_id")},
            )
            for hit in hits
        ]
        context = "\n\n".join(f"[{source.id}] {source.snippet}" for source in sources)
        prompt = (
            "Answer the question using only the retrieved paper evidence.\n"
            "Cite the paper ids inline when relevant.\n\n"
            f"Question: {payload.question}\n\n"
            f"Retrieved context:\n{context}"
        )
        generation_start = time.perf_counter()
        answer, prompt_tokens, completion_tokens = await self.llm_client.complete(prompt)
        generation_ms = (time.perf_counter() - generation_start) * 1000
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
            sources=sources,
            retrieval_info=RetrievalInfo(
                mode="vector",
                top_k=top_k,
                chunk_strategy=self.settings.chunking.strategy,
            ),
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            estimated_cost=cost,
        )
