#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import arxiv
import fitz
from tqdm import tqdm

from app.core.config import get_settings
from app.utils.tokens import count_tokens

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("download_arxiv_papers")


def clean_text(text: str) -> str:
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def pdf_to_text(pdf_path: Path) -> str:
    document = fitz.open(pdf_path)
    return clean_text("\n".join(page.get_text("text") for page in document))


def safe_id(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


def main() -> None:
    settings = get_settings()
    raw_dir = Path(settings.paths.raw_pdfs_dir)
    parsed_dir = Path(settings.paths.parsed_text_dir)
    jsonl_dir = Path(settings.paths.jsonl_dir)
    metadata_dir = Path(settings.paths.metadata_dir)
    for directory in (raw_dir, parsed_dir, jsonl_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    records_path = jsonl_dir / "papers.jsonl"
    seen_ids = set()
    total_tokens = 0
    existing_records = 0
    if records_path.exists():
        with records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                seen_ids.add(record["id"])
                total_tokens += int(record.get("token_count", 0))
                existing_records += 1
        logger.info("Resuming from %s existing records and %s tokens", existing_records, total_tokens)

    search = arxiv.Search(
        query=settings.dataset.query,
        max_results=settings.dataset.max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)

    with records_path.open("a", encoding="utf-8") as sink:
        for result in tqdm(client.results(search), total=settings.dataset.max_results, desc="arXiv papers"):
            paper_id = safe_id(result.get_short_id())
            if paper_id in seen_ids:
                continue

            pdf_path = raw_dir / f"{paper_id}.pdf"
            text_path = parsed_dir / f"{paper_id}.txt"
            try:
                if not pdf_path.exists():
                    result.download_pdf(dirpath=str(raw_dir), filename=pdf_path.name)
                    time.sleep(settings.dataset.request_delay_seconds)

                text = pdf_to_text(pdf_path)
                token_count = count_tokens(text)
                if token_count < settings.dataset.min_tokens_per_paper:
                    logger.warning("Skipping %s because it only has %s tokens", paper_id, token_count)
                    continue

                text_path.write_text(text, encoding="utf-8")
                record = {
                    "id": paper_id,
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "summary": result.summary,
                    "published": result.published.isoformat(),
                    "updated": result.updated.isoformat(),
                    "categories": result.categories,
                    "pdf_url": result.pdf_url,
                    "entry_id": result.entry_id,
                    "token_count": token_count,
                    "pdf_path": str(pdf_path),
                    "text_path": str(text_path),
                    "text": text,
                }
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                seen_ids.add(paper_id)
                total_tokens += token_count
                if total_tokens >= settings.dataset.target_tokens:
                    logger.info("Reached target token count: %s", total_tokens)
                    break
            except Exception as exc:
                logger.exception("Failed to process %s: %s", paper_id, exc)

    summary = {
        "num_papers": len(seen_ids),
        "total_tokens": total_tokens,
        "query": settings.dataset.query,
        "target_tokens": settings.dataset.target_tokens,
    }
    (metadata_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Dataset summary: %s", summary)


if __name__ == "__main__":
    main()
