#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PARSED_TEXT_DIR = REPO_ROOT / "data" / "parsed_text"
DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "datasets" / "local_arxiv_eval_60.json"

PRIORITY_IDS = [
    "1706.03762v7",
    "1810.04805v2",
    "2004.04906v3",
    "2005.11401v4",
    "2005.14165v4",
    "2205.14135v2",
    "2302.13971v1",
    "2307.09288v2",
    "2310.06825v1",
    "2401.04088v1",
    "2407.10671v4",
    "2412.15115v2",
    "2605.07847v1",
]

TITLE_SKIP_PREFIXES = (
    "provided proper attribution",
    "reproduce the tables",
    "preprint",
    "code:",
    "webpage:",
    "figure ",
    "table ",
    "arxiv:",
)

TITLE_SKIP_TOKENS = (
    "@",
    "university",
    "research",
    "google brain",
    "google research",
    "microsoft",
    "facebook ai",
    "abstract",
)


def main() -> None:
    args = parse_args()
    records = build_dataset(PARSED_TEXT_DIR, count=args.count)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} questions to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a local evaluation dataset from parsed arXiv paper text files.")
    parser.add_argument("--count", type=int, default=60, help="Number of question-answer pairs to generate.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON path.")
    return parser.parse_args()


def build_dataset(parsed_text_dir: Path, *, count: int) -> list[dict[str, object]]:
    papers = []
    for path in sorted(parsed_text_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        title = extract_title(text, path.stem)
        abstract = extract_abstract(text)
        if not title or not abstract:
            continue
        if len(abstract.split()) < 80:
            continue
        papers.append(
            {
                "id": path.stem,
                "title": title,
                "abstract": abstract,
                "path": path,
            }
        )

    papers.sort(key=paper_sort_key)
    selected = papers[:count]
    return [paper_to_question(record, index + 1) for index, record in enumerate(selected)]


def paper_sort_key(paper: dict[str, object]) -> tuple[int, str]:
    paper_id = str(paper["id"])
    try:
        priority_index = PRIORITY_IDS.index(paper_id)
    except ValueError:
        priority_index = len(PRIORITY_IDS) + 1
    return (priority_index, paper_id)


def extract_title(text: str, fallback_id: str) -> str:
    lines = [normalize_space(line) for line in text.splitlines()]
    try:
        abstract_idx = next(i for i, line in enumerate(lines) if line.lower() == "abstract")
    except StopIteration:
        abstract_idx = min(len(lines), 40)

    title_lines: list[str] = []
    for line in lines[:abstract_idx]:
        if not line:
            if title_lines:
                break
            continue
        lowered = line.lower()
        if lowered.startswith(TITLE_SKIP_PREFIXES):
            continue
        if is_author_like(line):
            if title_lines:
                break
            continue
        if any(token in lowered for token in TITLE_SKIP_TOKENS):
            if title_lines:
                break
            continue
        if len(line) < 4 or len(line) > 200:
            if title_lines:
                break
            continue
        title_lines.append(line)
        if len(" ".join(title_lines)) > 180:
            break
    if title_lines:
        return normalize_space(" ".join(title_lines))
    return fallback_id


def is_author_like(line: str) -> bool:
    lowered = line.lower()
    if "@" in line:
        return True
    if line.count(",") >= 2:
        return True
    if any(keyword in lowered for keyword in ("university", "research", "laboratory", "institute", "facebook ai", "google ai")):
        return True
    words = line.split()
    if len(words) >= 4 and all(re.fullmatch(r"[A-Z][A-Za-z.\-’']+", word.strip(",;:†‡⋆*")) for word in words[: min(len(words), 6)]):
        return True
    return False


def extract_abstract(text: str) -> str:
    normalized = text.replace("\r\n", "\n")
    match = re.search(
        r"\bAbstract\b\s*(.*?)\n\s*(?:\d+\s*\n\s*[A-Z][^\n]{1,120}|\d+\s+[A-Z][^\n]{1,120}|Introduction\b)",
        normalized,
        flags=re.DOTALL,
    )
    if not match:
        return ""

    abstract = normalize_space(match.group(1))
    abstract = re.sub(r"\[[0-9,\s]+\]", "", abstract)
    abstract = re.sub(r"\s+", " ", abstract).strip()
    return clip_sentences(abstract, max_sentences=4, max_chars=850)


def clip_sentences(text: str, *, max_sentences: int, max_chars: int) -> str:
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text) if segment.strip()]
    if not sentences:
        return text[:max_chars].strip()

    clipped = " ".join(sentences[:max_sentences]).strip()
    if len(clipped) <= max_chars:
        return clipped
    return clipped[: max_chars - 1].rstrip() + "…"


def paper_to_question(paper: dict[str, object], ordinal: int) -> dict[str, object]:
    paper_id = str(paper["id"])
    title = str(paper["title"])
    abstract = str(paper["abstract"])

    templates = [
        (
            f"What does the paper with arXiv ID {paper_id} propose or study?",
            "lookup",
            "easy",
        ),
        (
            f"What is the main contribution or claim of the paper with arXiv ID {paper_id}, according to its abstract?",
            "summarization",
            "medium",
        ),
        (
            f"What problem does the paper with arXiv ID {paper_id} address, and what approach does it take?",
            "summarization",
            "medium",
        ),
    ]
    question, category, difficulty = templates[(ordinal - 1) % len(templates)]
    return {
        "id": f"local-arxiv-{ordinal:03d}",
        "question": question,
        "ground_truth": abstract,
        "category": category,
        "difficulty": difficulty,
        "sources": [paper_id],
        "metadata": {
            "title": title,
            "paper_id": paper_id,
            "generator": "generate_local_eval_dataset.py",
        },
    }


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


if __name__ == "__main__":
    main()
