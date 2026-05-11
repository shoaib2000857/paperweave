#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json

from app.core.dependencies import build_container
from app.models.api import AskRequest


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()
    container = build_container()
    response = await container.basic_rag_pipeline.run(AskRequest(question=args.question, top_k=args.top_k))
    print(json.dumps(response.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
