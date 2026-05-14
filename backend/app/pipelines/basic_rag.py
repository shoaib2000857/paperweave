from __future__ import annotations

import time
import logging

from app.core.config import Settings
from app.models.api import AskRequest, RetrievalInfo, SourceRecord
from app.pipelines.base import BasePipeline
from app.services.llm import LLMClient
from app.storage.dataset_store import DatasetStore
from app.storage.rag_store import BasicRAGStore

logger = logging.getLogger(__name__)


class BasicRAGPipeline(BasePipeline):
    pipeline_name = "basic-rag"

    def __init__(
        self,
        settings: Settings,
        llm_client: LLMClient,
        embedding_provider,
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
        hits = self.rag_store.search(payload.question, top_k=top_k)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        logger.info("Basic RAG query=%r top_k=%s hits=%s", payload.question, top_k, len(hits))

        sources = [
            SourceRecord(
                id=str(hit.get("chunk_id") or hit.get("id") or "unknown"),
                title=hit.get("title") or hit.get("paper_filename") or hit.get("source"),
                snippet=hit["text"][:600],
                score=hit.get("score"),
                metadata={
                    "source": hit.get("source"),
                    "paper_filename": hit.get("paper_filename") or hit.get("source"),
                    "page": hit.get("page"),
                    "chunk_id": hit.get("chunk_id"),
                },
            )
            for hit in hits
        ]

        if not sources:
            status = self.rag_store.corpus_status()
            if not any(
                [
                    status["raw_pdf_count"],
                    status["parsed_text_count"],
                    status["parsed_markdown_count"],
                    status["jsonl_exists"],
                ]
            ):
                diagnostic = (
                    "Basic RAG has no local paper corpus to index. Add PDFs to data/raw_pdfs, parsed text to "
                    "data/parsed_text, markdown to data/parsed_markdown, or run "
                    "`python scripts/build_basic_rag.py --bootstrap-arxiv` to create a small public arXiv corpus. "
                    "Then run `python scripts/test_basic_rag.py \""
                    f"{payload.question}\"`."
                )
            else:
                diagnostic = (
                    "Basic RAG found local corpus files but no retrieved Chroma chunks. Rebuild the index with "
                    "`python scripts/build_basic_rag.py` and verify it with "
                    f"`python scripts/test_basic_rag.py \"{payload.question}\"`."
                )
            latency_ms = (time.perf_counter() - started) * 1000
            return self._build_response(
                answer=diagnostic,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=latency_ms,
                sources=[],
                retrieval_info=RetrievalInfo(
                    mode="vector",
                    top_k=top_k,
                    chunk_strategy="recursive_character",
                    raw={
                        "diagnostic": "zero_retrieved_chunks",
                        "chroma_dir": self.settings.paths.basic_rag_dir,
                        "embedding_model": self.settings.providers.embeddings.model,
                        "corpus_status": status,
                    },
                ),
                retrieval_ms=retrieval_ms,
                generation_ms=0.0,
                estimated_cost=0.0,
                raw={"diagnostic": "zero_retrieved_chunks"},
            )

        context = "\n\n".join(
            (
                f"[{index}] source={source.metadata.get('paper_filename') or source.id}; "
                f"page={source.metadata.get('page')}; chunk={source.metadata.get('chunk_id')}\n"
                f"{source.snippet}"
            )
            for index, source in enumerate(sources, start=1)
        )
        prompt = (
            "You are PaperWeave's conventional vector RAG QA system for scientific papers.\n"
            "Answer ONLY from the retrieved context below. Do not use outside knowledge.\n"
            "If the context is insufficient, say what is missing or uncertain.\n"
            "Use a concise scientific QA tone and cite sources inline as [1], [2], etc.\n\n"
            f"Question: {payload.question}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Grounded answer:"
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
                chunk_strategy="recursive_character",
                raw={
                    "chroma_dir": self.settings.paths.basic_rag_dir,
                    "embedding_model": self.settings.providers.embeddings.model,
                    "retrieved_chunks": len(sources),
                },
            ),
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            estimated_cost=cost,
        )
