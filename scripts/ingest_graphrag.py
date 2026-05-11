#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from app.core.config import get_settings


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--container-source-dir", default=None)
    parser.add_argument("--mode", choices=["pdf", "markdown", "text"], default="pdf")
    parser.add_argument("--file-format", choices=["multi", "json"], default=None)
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
    async with httpx.AsyncClient(timeout=300.0, auth=(settings.tigergraph.username, settings.tigergraph.password)) as client:
        create_payload = {
            "data_source": "server",
            "data_source_config": {"data_path": container_source_dir},
            "file_format": file_format,
            "loader_config": {},
        }
        create_response = await client.post(
            f"{settings.graphrag.api_base.rstrip('/')}/{settings.graphrag.graph_name}/graphrag/create_ingest",
            json=create_payload,
        )
        create_response.raise_for_status()
        created = create_response.json()

        ingest_payload = {
            "load_job_id": created["load_job_id"],
            "data_source_id": created["data_source_id"],
            "file_path": created.get("data_path", container_source_dir),
        }
        ingest_response = await client.post(
            f"{settings.graphrag.api_base.rstrip('/')}/{settings.graphrag.graph_name}/graphrag/ingest",
            json=ingest_payload,
        )
        ingest_response.raise_for_status()
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
