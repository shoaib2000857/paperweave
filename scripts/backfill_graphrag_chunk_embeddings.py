#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import httpx
from pyTigerGraph import TigerGraphConnection

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "backend"))

from app.core.config import get_settings  # noqa: E402


def _normalize_host(value: str | None) -> str:
    if not value:
        return "http://127.0.0.1"
    host = value.rstrip("/")
    if host == "http://tigergraph":
        return "http://127.0.0.1"
    return host


def _extract_vertices(payload: list[dict]) -> list[str]:
    if not payload:
        return []
    results = payload[0].get("result", [])
    return [item["v_id"] for item in results if "v_id" in item]


def _extract_chunk_text(payload: list[dict]) -> str:
    if not payload:
        return ""
    chunk_content = payload[0].get("ChunkContent", [])
    if not chunk_content:
        return ""
    return chunk_content[0]["attributes"]["text"]


def _embed_text(client: httpx.Client, base_url: str, model: str, text: str) -> list[float]:
    response = client.post(
        f"{base_url.rstrip('/')}/api/embed",
        json={"model": model, "input": text},
        timeout=300.0,
    )
    response.raise_for_status()
    payload = response.json()
    if "embeddings" in payload and payload["embeddings"]:
        return payload["embeddings"][0]
    if "embedding" in payload:
        return payload["embedding"]
    raise RuntimeError(f"Unexpected Ollama embed response shape: {payload}")


def _batched(values: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-name", default=None)
    parser.add_argument("--tg-host", default=None)
    parser.add_argument("--tg-port", type=int, default=14240)
    parser.add_argument("--embedding-base-url", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    graph_name = args.graph_name or settings.graphrag.graph_name
    tg_host = _normalize_host(args.tg_host or settings.tigergraph.hostname)
    embedding_base_url = args.embedding_base_url or settings.providers.embeddings.base_url or "http://127.0.0.1:11434"
    embedding_model = args.embedding_model or settings.providers.embeddings.model

    conn = TigerGraphConnection(
        host=tg_host,
        graphname=graph_name,
        username=settings.tigergraph.username,
        password=settings.tigergraph.password,
        restppPort=args.tg_port,
        gsPort=args.tg_port,
    )

    vertices_payload = conn.runInstalledQuery(
        "get_vertices_or_remove",
        params={
            "v_type": "DocumentChunk",
            "keyword": "",
            "with_edge": "false",
            "remove": False,
        },
    )
    chunk_ids = _extract_vertices(vertices_payload)
    if args.limit is not None:
        chunk_ids = chunk_ids[: args.limit]
    if not chunk_ids:
        raise SystemExit("No DocumentChunk vertices found")

    processed = 0
    with httpx.Client() as client:
        for batch_ids in _batched(chunk_ids, args.batch_size):
            vertices: dict[str, dict[str, dict[str, list[float]]]] = {"DocumentChunk": {}}
            for chunk_id in batch_ids:
                chunk_payload = conn.runInstalledQuery("StreamChunkContent", params={"chunk": chunk_id})
                chunk_text = _extract_chunk_text(chunk_payload)
                if not chunk_text:
                    continue
                embedding = _embed_text(client, embedding_base_url, embedding_model, chunk_text)
                vertices["DocumentChunk"][chunk_id] = {"embedding": {"value": embedding}}

            if not vertices["DocumentChunk"]:
                continue

            conn.upsertData(json.dumps({"vertices": vertices}))
            processed += len(vertices["DocumentChunk"])
            print(json.dumps({"processed": processed, "last_batch": len(vertices["DocumentChunk"])}))

    print(
        json.dumps(
            {
                "graph_name": graph_name,
                "tg_host": tg_host,
                "tg_port": args.tg_port,
                "embedding_base_url": embedding_base_url,
                "embedding_model": embedding_model,
                "requested_chunks": len(chunk_ids),
                "embedded_chunks": processed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
