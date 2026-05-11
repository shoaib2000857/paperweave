from __future__ import annotations

import time
from typing import Any

import httpx

from app.core.config import Settings
from app.models.api import AskRequest, RetrievalInfo, SourceRecord
from app.pipelines.base import BasePipeline
from app.services.llm import LLMClient


class GraphRAGPipeline(BasePipeline):
    pipeline_name = "graphrag"

    def __init__(self, settings: Settings, llm_client: LLMClient):
        self.settings = settings
        self.llm_client = llm_client

    async def run(self, payload: AskRequest):
        started = time.perf_counter()
        retrieval_start = time.perf_counter()
        top_k = payload.top_k or self.settings.graphrag.top_k
        num_hops = payload.num_hops or self.settings.graphrag.num_hops
        request_body = {
            "question": payload.question,
            "method": "hybrid",
            "method_params": {
                "top_k": top_k,
                "num_hops": num_hops,
                "num_seen_min": 1,
                "indices": ["DocumentChunk"],
                "chunk_only": self.settings.graphrag.chunk_only,
                "doc_only": self.settings.graphrag.doc_only,
                "verbose": True,
            },
        }
        endpoint = f"{self.settings.graphrag.api_base.rstrip('/')}/{self.settings.graphrag.graph_name}/graphrag/answerquestion"
        try:
            timeout = httpx.Timeout(180.0, connect=2.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    endpoint,
                    json=request_body,
                    auth=(self.settings.tigergraph.username, self.settings.tigergraph.password),
                )
                response.raise_for_status()
                raw = response.json()
        except httpx.ConnectError:
            return self._build_unavailable_response(
                started=started,
                retrieval_start=retrieval_start,
                top_k=top_k,
                num_hops=num_hops,
                endpoint=endpoint,
                message=(
                    "GraphRAG service is not reachable. Start the TigerGraph GraphRAG API "
                    f"or set GRAPHRAG_API_BASE to the running service. Tried: {endpoint}"
                ),
            )
        except httpx.TimeoutException:
            return self._build_unavailable_response(
                started=started,
                retrieval_start=retrieval_start,
                top_k=top_k,
                num_hops=num_hops,
                endpoint=endpoint,
                message=f"GraphRAG service timed out after 180 seconds. Tried: {endpoint}",
            )
        except httpx.HTTPStatusError as exc:
            return self._build_unavailable_response(
                started=started,
                retrieval_start=retrieval_start,
                top_k=top_k,
                num_hops=num_hops,
                endpoint=endpoint,
                message=f"GraphRAG service returned HTTP {exc.response.status_code}: {exc.response.text[:500]}",
            )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        answer = raw.get("response") or raw.get("natural_language_response", "")
        sources = self._extract_sources(raw)
        prompt_tokens = len(payload.question.split())
        completion_tokens = len(answer.split())
        generation_ms = max((time.perf_counter() - started) * 1000 - retrieval_ms, 0.0)
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
                mode="graph",
                top_k=top_k,
                num_hops=num_hops,
                chunk_strategy=self.settings.graphrag.chunker,
                graph_name=self.settings.graphrag.graph_name,
                raw={
                    "retrieved": raw.get("retrieved", []),
                    "verbose": raw.get("verbose", {}),
                },
            ),
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            estimated_cost=cost,
            raw=raw,
        )

    def _build_unavailable_response(
        self,
        started: float,
        retrieval_start: float,
        top_k: int,
        num_hops: int,
        endpoint: str,
        message: str,
    ):
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        latency_ms = (time.perf_counter() - started) * 1000
        return self._build_response(
            answer=message,
            prompt_tokens=len(message.split()),
            completion_tokens=0,
            latency_ms=latency_ms,
            sources=[],
            retrieval_info=RetrievalInfo(
                mode="graph_unavailable",
                top_k=top_k,
                num_hops=num_hops,
                chunk_strategy=self.settings.graphrag.chunker,
                graph_name=self.settings.graphrag.graph_name,
                raw={"endpoint": endpoint, "error": message},
            ),
            retrieval_ms=retrieval_ms,
            generation_ms=0.0,
            estimated_cost=0.0,
            raw={"status": "unavailable", "endpoint": endpoint, "error": message},
        )

    def _extract_sources(self, raw: dict[str, Any]) -> list[SourceRecord]:
        sources: list[SourceRecord] = []

        for idx, item in enumerate(raw.get("retrieved", []) or [], start=1):
            candidate_answer = item.get("candidate_answer", "")
            if not candidate_answer:
                continue
            sources.append(
                SourceRecord(
                    id=f"retrieved-{idx}",
                    title="GraphRAG Candidate",
                    snippet=candidate_answer.strip(),
                    score=item.get("score"),
                    metadata=item,
                )
            )

        verbose = raw.get("verbose", {}) or {}
        final_retrieval = verbose.get("final_retrieval", {}) if isinstance(verbose, dict) else {}
        for vertex_id, snippets in final_retrieval.items():
            text_snippets = [snippet for snippet in snippets if snippet]
            if not text_snippets:
                continue
            sources.append(
                SourceRecord(
                    id=str(vertex_id),
                    title=str(vertex_id),
                    snippet="\n\n".join(text_snippets).strip(),
                    score=None,
                    metadata={"vertex_id": vertex_id, "snippets": snippets},
                )
            )

        deduped: list[SourceRecord] = []
        seen: set[str] = set()
        for source in sources:
            if source.id in seen:
                continue
            seen.add(source.id)
            deduped.append(source)
        return deduped
