#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json

from app.core.dependencies import build_container
from app.models.api import BenchmarkRequest


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", default=None)
    args = parser.parse_args()
    container = build_container()
    response = await container.benchmark_service.run(BenchmarkRequest(question_file=args.question_file))
    print(json.dumps(response.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
