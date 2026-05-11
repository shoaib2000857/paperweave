#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from app.core.config import get_settings
from app.services.providers import LLMProviderFactory
from app.storage.rag_store import BasicRAGStore
from app.utils.chunking import sliding_window_chunks


def main() -> None:
    settings = get_settings()
    records_path = Path(settings.paths.jsonl_dir) / "papers.jsonl"
    papers = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    embedding_provider = LLMProviderFactory(settings).build_embedding_provider()
    rag_store = BasicRAGStore(settings)

    docs = []
    for paper in papers:
        for chunk in sliding_window_chunks(paper["text"], settings.chunking.chunk_size, settings.chunking.overlap):
            docs.append(
                {
                    "id": f'{paper["id"]}:{chunk.index}',
                    "paper_id": paper["id"],
                    "title": paper["title"],
                    "chunk_id": chunk.index,
                    "text": chunk.text,
                }
            )

    embeddings = embedding_provider.embed_documents([doc["text"] for doc in tqdm(docs, desc="Embed chunks")])
    rag_store.save(np.array(embeddings, dtype="float32"), docs)
    meta = {"num_papers": len(papers), "num_chunks": len(docs), "embedding_model": settings.providers.embeddings.model}
    (Path(settings.paths.basic_rag_dir) / "build_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
