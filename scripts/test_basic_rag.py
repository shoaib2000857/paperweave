#!/usr/bin/env python3
from __future__ import annotations

import argparse

from app.core.config import get_settings
from app.storage.rag_store import BasicRAGStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the persisted PaperWeave Basic RAG Chroma index.")
    parser.add_argument("question", nargs="?", default="What does the paper corpus say about retrieval augmented generation?")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    settings = get_settings()
    store = BasicRAGStore(settings)
    print(f"Chroma directory: {settings.paths.basic_rag_dir}")
    print(f"Collection: {BasicRAGStore.collection_name}")
    print(f"Stored chunks: {store.count()}")

    hits = store.search(args.question, top_k=args.top_k)
    print(f"Retrieved chunks: {len(hits)}")
    if not hits:
        raise SystemExit("No chunks retrieved. Rebuild with `python scripts/build_basic_rag.py` and check Ollama embeddings.")

    for index, hit in enumerate(hits, start=1):
        print("\n" + "=" * 80)
        print(f"Hit {index}")
        print(f"Score: {hit.get('score')}")
        print(f"Source: {hit.get('paper_filename') or hit.get('source')}")
        print(f"Page: {hit.get('page')}")
        print(f"Chunk: {hit.get('chunk_id')}")
        print("-" * 80)
        print(str(hit.get("text", ""))[:1200])


if __name__ == "__main__":
    main()
