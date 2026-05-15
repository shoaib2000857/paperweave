#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from app.core.config import get_settings


async def post_json_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    payload: dict,
    max_attempts: int,
    backoff_seconds: float,
) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError) as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            await asyncio.sleep(backoff_seconds * attempt)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"POST {url} failed without a captured exception")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--container-source-dir", default=None)
    parser.add_argument("--mode", choices=["pdf", "markdown", "text"], default="pdf")
    parser.add_argument("--file-format", choices=["multi", "json"], default=None)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    settings = get_settings()
    local_source_dir = Path(args.source_dir or {
        "pdf": settings.paths.raw_pdfs_dir,
        "markdown": settings.paths.parsed_markdown_dir,
        "text": settings.paths.parsed_text_dir,
    }[args.mode])
    pattern = {"pdf": "*.pdf", "markdown": "*.md", "text": "*.txt"}[args.mode]
    files = sorted(local_source_dir.glob(pattern))
    if not files:
        raise SystemExit(f"No input files found in {local_source_dir} matching {pattern}")

    container_source_dir = args.container_source_dir or {
        "pdf": "/paperweave-data/raw_pdfs",
        "markdown": "/paperweave-data/parsed_markdown",
        "text": "/paperweave-data/parsed_text",
    }[args.mode]
    file_format = args.file_format or ("multi" if args.mode == "pdf" else "json")
    timeout = httpx.Timeout(args.timeout_seconds, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout, auth=(settings.tigergraph.username, settings.tigergraph.password)) as client:
        create_payload = {
            "data_source": "server",
            "data_source_config": {"data_path": container_source_dir},
            "file_format": file_format,
            "loader_config": {},
        }
        create_response = await post_json_with_retry(
            client,
            f"{settings.graphrag.api_base.rstrip('/')}/{settings.graphrag.graph_name}/graphrag/create_ingest",
            payload=create_payload,
            max_attempts=args.max_attempts,
            backoff_seconds=5.0,
        )
        created = create_response.json()

        ingest_payload = {
            "load_job_id": created["load_job_id"],
            "data_source_id": created["data_source_id"],
            "file_path": created.get("data_path", container_source_dir),
        }
        ingest_response = await post_json_with_retry(
            client,
            f"{settings.graphrag.api_base.rstrip('/')}/{settings.graphrag.graph_name}/graphrag/ingest",
            payload=ingest_payload,
            max_attempts=args.max_attempts,
            backoff_seconds=10.0,
        )
        print(
            json.dumps(
                {
                    "local_source_dir": str(local_source_dir),
                    "container_source_dir": container_source_dir,
                    "file_count": len(files),
                    "create_ingest": created,
                    "ingest_result": ingest_response.json(),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
