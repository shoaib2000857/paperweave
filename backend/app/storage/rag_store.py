from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb

try:
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover - compatibility for older LangChain installs
    from langchain_community.vectorstores import Chroma

try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:  # pragma: no cover - compatibility for older LangChain installs
    from langchain_community.embeddings import OllamaEmbeddings

from app.core.config import Settings

logger = logging.getLogger(__name__)


class BasicRAGStore:
    collection_name = "paperweave_basic_rag"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_dir = Path(settings.paths.basic_rag_dir)
        self.embedding_model = settings.providers.embeddings.model
        self.embedding_base_url = settings.providers.embeddings.base_url or "http://localhost:11434"

    def _embeddings(self) -> OllamaEmbeddings:
        return OllamaEmbeddings(model=self.embedding_model, base_url=self.embedding_base_url)

    def _store(self) -> Chroma:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embeddings(),
            persist_directory=str(self.base_dir),
        )

    def is_ready(self) -> bool:
        if not self.base_dir.exists():
            return False
        return self.count() > 0

    def count(self) -> int:
        if not self.base_dir.exists():
            return 0
        try:
            client = chromadb.PersistentClient(path=str(self.base_dir))
            collection = client.get_collection(self.collection_name)
            return int(collection.count())
        except Exception as exc:
            logger.warning("Basic RAG Chroma collection count failed: %s", exc)
            return 0

    def corpus_status(self) -> dict[str, Any]:
        paths = {
            "raw_pdfs_dir": Path(self.settings.paths.raw_pdfs_dir),
            "parsed_text_dir": Path(self.settings.paths.parsed_text_dir),
            "parsed_markdown_dir": Path(self.settings.paths.parsed_markdown_dir),
            "jsonl_path": Path(self.settings.paths.jsonl_dir) / "papers.jsonl",
        }
        return {
            "chroma_dir": str(self.base_dir),
            "stored_chunks": self.count(),
            "raw_pdf_count": sum(1 for _ in paths["raw_pdfs_dir"].glob("*.pdf")) if paths["raw_pdfs_dir"].exists() else 0,
            "parsed_text_count": sum(1 for _ in paths["parsed_text_dir"].glob("**/*.txt")) if paths["parsed_text_dir"].exists() else 0,
            "parsed_markdown_count": (
                sum(1 for _ in paths["parsed_markdown_dir"].glob("**/*.md")) if paths["parsed_markdown_dir"].exists() else 0
            ),
            "jsonl_exists": paths["jsonl_path"].exists(),
            "jsonl_path": str(paths["jsonl_path"]),
        }

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if not self.is_ready():
            logger.warning("Basic RAG index is empty or missing at %s", self.base_dir)
            return []
        logger.info("Searching Basic RAG Chroma index path=%s top_k=%s", self.base_dir, top_k)
        hits = self._store().similarity_search_with_relevance_scores(query, k=top_k)
        results: list[dict[str, Any]] = []
        for document, score in hits:
            metadata = dict(document.metadata or {})
            results.append(
                {
                    "id": metadata.get("chunk_id") or getattr(document, "id", None) or metadata.get("source") or "unknown",
                    "text": document.page_content,
                    "score": float(score),
                    **metadata,
                }
            )
        logger.info("Basic RAG retrieved %s chunks", len(results))
        return results
