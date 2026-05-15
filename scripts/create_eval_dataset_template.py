#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.dataset import write_dataset_template


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a PaperWeave evaluation dataset template.")
    parser.add_argument(
        "--output",
        default="evaluation/datasets/scientific_qa_template.json",
        help="Path for the JSON dataset template.",
    )
    parser.add_argument("--count", type=int, default=5, help="Number of placeholder questions to include.")
    args = parser.parse_args()
    write_dataset_template(args.output, count=args.count)
    print(f"Wrote evaluation dataset template to {args.output}")


if __name__ == "__main__":
    main()
