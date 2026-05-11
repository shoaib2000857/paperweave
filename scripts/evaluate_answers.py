#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.core.dependencies import build_container


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("answer")
    parser.add_argument("reference_answer")
    args = parser.parse_args()
    container = build_container()
    result = await container.evaluation_service.evaluate(args.question, args.answer, args.reference_answer)
    print(json.dumps(result.model_dump(mode="json") if result else {}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
