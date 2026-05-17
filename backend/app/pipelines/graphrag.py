"""
Cloud-based GraphRAG pipeline.

Replaces the previous localhost TigerGraph microservice call with a
stable, cloud-based flow:

  1. Retrieve top-k chunks from the existing Chroma vector store.
  2. Pass chunks to CloudGraphRAGService which:
       a. Extracts entities + relationships via Gemini.
       b. Queries TigerGraph Cloud (if credentials provided) or
          a NetworkX in-memory graph as fallback.
       c. Enriches context and generates the final answer via Gemini.

Root-cause note:
  The previous implementation called http://localhost:8000 (Dockerized
  TigerGraph GraphRAG microservice). That service is not running in most
  demo / hackathon environments, causing ConnectError → unavailable
  placeholder answers and breaking the entire /ask/all endpoint when
  asyncio.gather() propagated unrelated evaluation failures. This
  pipeline is self-contained and crash-resistant.
"""

from __future__ import annotations

import time
from typing import Any

from app.core.config import Settings
from app.models.api import AskRequest, RetrievalInfo, SourceRecord
from app.pipelines.base import BasePipeline
from app.services.cloud_graphrag import CloudGraphRAGService, build_cloud_graphrag_service
from app.services.llm import LLMClient
from app.storage.rag_store import BasicRAGStore


class GraphRAGPipeline(BasePipeline):
    pipeline_name = "graphrag"

    def __init__(
        self,
        settings: Settings,
        llm_client: LLMClient,
        rag_store: BasicRAGStore,
    ):
        self.settings = settings
        self.llm_client = llm_client
        self.rag_store = rag_store
        self._cloud: CloudGraphRAGService = build_cloud_graphrag_service(
            graph_name=self.settings.graphrag.graph_name
        )

    async def run(self, payload: AskRequest):
        started = time.perf_counter()
        top_k = payload.top_k or self.settings.graphrag.top_k
        num_hops = payload.num_hops or self.settings.graphrag.num_hops

        # --- 1. Retrieve chunks from Chroma ---
        retrieval_start = time.perf_counter()
        try:
            raw_chunks = self.rag_store.search(payload.question, top_k=top_k)
        except Exception as exc:
            raw_chunks = []
            retrieval_error = str(exc)
        else:
            retrieval_error = None
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        # --- 2. Run cloud GraphRAG ---
        generation_start = time.perf_counter()
        try:
            result = await self._cloud.answer(
                question=payload.question,
                chunks=raw_chunks,
                top_k=top_k,
            )
        except Exception as exc:
            generation_ms = (time.perf_counter() - generation_start) * 1000
            latency_ms = (time.perf_counter() - started) * 1000
            return self._build_unavailable_response(
                started_ts=started,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                latency_ms=latency_ms,
                top_k=top_k,
                num_hops=num_hops,
                message=f"Cloud GraphRAG failed: {exc}",
            )
        generation_ms = (time.perf_counter() - generation_start) * 1000
        latency_ms = (time.perf_counter() - started) * 1000

        # --- 3. Build answer text ---
        answer = result["answer"]
        reasoning = result.get("reasoning", "")
        if reasoning:
            answer = f"{answer}\n\n---\n**Graph Reasoning:** {reasoning}"

        # --- 4. Build sources from chunks + graph metadata ---
        sources = self._build_sources(raw_chunks, result)

        prompt_tokens = len(payload.question.split()) + sum(
            len(c.get("text", "").split()) for c in raw_chunks
        )
        completion_tokens = len(answer.split())
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
                mode="cloud_graph",
                top_k=top_k,
                num_hops=num_hops,
                chunk_strategy=self.settings.graphrag.chunker,
                graph_name=self.settings.graphrag.graph_name,
                raw={
                    "graph_nodes": result.get("graph_nodes", []),
                    "graph_edges": result.get("graph_edges", []),
                    "entity_count": len(result.get("entities", [])),
                    "relationship_count": len(result.get("relationships", [])),
                    "retrieved_chunks": len(raw_chunks),
                    "retrieval_error": retrieval_error,
                    "graph_backend": (
                        "tigergraph_cloud"
                        if (self._cloud.tigergraph_url and self._cloud.tigergraph_api_key)
                        else "networkx_fallback"
                    ),
                },
            ),
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            estimated_cost=cost,
            raw=result,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_sources(
        self, raw_chunks: list[dict[str, Any]], result: dict[str, Any]
    ) -> list[SourceRecord]:
        sources: list[SourceRecord] = []
        seen: set[str] = set()

        # Chunks from Chroma
        for idx, chunk in enumerate(raw_chunks, start=1):
            chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or f"chunk-{idx}")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            sources.append(
                SourceRecord(
                    id=chunk_id,
                    title=chunk.get("title") or chunk.get("paper_filename") or chunk.get("source"),
                    snippet=(chunk.get("text") or chunk.get("snippet") or "")[:600],
                    score=chunk.get("score"),
                    metadata={
                        "source": chunk.get("source"),
                        "paper_filename": chunk.get("paper_filename") or chunk.get("source"),
                        "page": chunk.get("page"),
                        "chunk_id": chunk.get("chunk_id"),
                    },
                )
            )

        # Graph nodes as additional "sources" (graph evidence)
        for node in result.get("graph_nodes", [])[:5]:
            node_id = f"graph-node-{node.get('id', '')}"
            if node_id in seen or not node.get("description"):
                continue
            seen.add(node_id)
            sources.append(
                SourceRecord(
                    id=node_id,
                    title=f"Graph Entity: {node.get('id')}",
                    snippet=node.get("description", ""),
                    score=None,
                    metadata={"node_type": node.get("type"), "source": "graph"},
                )
            )

        return sources

    def _build_unavailable_response(
        self,
        started_ts: float,
        retrieval_ms: float,
        generation_ms: float,
        latency_ms: float,
        top_k: int,
        num_hops: int,
        message: str,
    ):
        return self._build_response(
            answer=message,
            prompt_tokens=len(message.split()),
            completion_tokens=0,
            latency_ms=latency_ms,
            sources=[],
            retrieval_info=RetrievalInfo(
                mode="cloud_graph_unavailable",
                top_k=top_k,
                num_hops=num_hops,
                chunk_strategy=self.settings.graphrag.chunker,
                graph_name=self.settings.graphrag.graph_name,
                raw={"error": message},
            ),
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            estimated_cost=0.0,
            raw={"status": "unavailable", "error": message},
        )
