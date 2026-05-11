from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.core.config import Settings


class BasicRAGStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_dir = Path(settings.paths.basic_rag_dir)
        self.index_path = self.base_dir / "index.faiss"
        self.docs_path = self.base_dir / "docs.pkl"
        self.meta_path = self.base_dir / "index_meta.json"

    def is_ready(self) -> bool:
        return self.index_path.exists() and self.docs_path.exists()

    def save(self, embeddings: np.ndarray, docs: list[dict[str, Any]]) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype("float32"))
        faiss.write_index(index, str(self.index_path))
        with self.docs_path.open("wb") as handle:
            pickle.dump(docs, handle)
        self.meta_path.write_text(json.dumps({"count": len(docs)}, indent=2), encoding="utf-8")

    def search(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        if not self.is_ready():
            return []
        index = faiss.read_index(str(self.index_path))
        with self.docs_path.open("rb") as handle:
            docs = pickle.load(handle)
        vector = np.array([query_embedding], dtype="float32")
        scores, indices = index.search(vector, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            results.append({**docs[idx], "score": float(score)})
        return results
