#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
from collections.abc import Iterable
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


def build_search(query: str, *, max_results: int, sort_by: arxiv.SortCriterion, sort_order: arxiv.SortOrder) -> arxiv.Search:
    return arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by,
        sort_order=sort_order,
    )


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def download_pdf_with_retries(
    result: arxiv.Result,
    *,
    pdf_path: Path,
    request_delay_seconds: float,
    max_attempts: int = 3,
) -> None:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        remove_if_exists(pdf_path)
        try:
            result.download_pdf(dirpath=str(pdf_path.parent), filename=pdf_path.name)
            with fitz.open(pdf_path):
                pass
            time.sleep(request_delay_seconds)
            return
        except (urllib.error.ContentTooShortError, fitz.FileDataError, RuntimeError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "Retrying PDF download for %s after attempt %s/%s failed: %s",
                result.get_short_id(),
                attempt,
                max_attempts,
                exc,
            )
            remove_if_exists(pdf_path)
            time.sleep(request_delay_seconds * attempt)
    if last_error is not None:
        raise last_error


def extract_text_from_pdf(
    *,
    result: arxiv.Result,
    pdf_path: Path,
    min_tokens_per_paper: int,
    request_delay_seconds: float,
) -> tuple[str, int]:
    needs_redownload = not pdf_path.exists()
    if not needs_redownload:
        try:
            text = pdf_to_text(pdf_path)
            token_count = count_tokens(text)
            if token_count > 0:
                return text, token_count
            logger.warning("Detected empty text in %s, forcing a fresh PDF download", pdf_path.name)
        except (urllib.error.ContentTooShortError, fitz.FileDataError, RuntimeError, ValueError) as exc:
            logger.warning("Detected unreadable PDF %s, forcing a fresh download: %s", pdf_path.name, exc)
        needs_redownload = True

    if needs_redownload:
        download_pdf_with_retries(
            result,
            pdf_path=pdf_path,
            request_delay_seconds=request_delay_seconds,
        )
        text = pdf_to_text(pdf_path)
        token_count = count_tokens(text)
        if token_count == 0:
            raise ValueError(f"Downloaded PDF {pdf_path.name} but extracted 0 tokens")
        return text, token_count

    text = pdf_to_text(pdf_path)
    token_count = count_tokens(text)
    if token_count < min_tokens_per_paper:
        return text, token_count
    return text, token_count


def iter_curated_id_results(
    client: arxiv.Client,
    arxiv_ids: list[str],
) -> Iterable[tuple[str, arxiv.Result | None]]:
    for arxiv_id in arxiv_ids:
        search = arxiv.Search(id_list=[arxiv_id], max_results=1)
        try:
            match = next(iter(client.results(search)), None)
        except arxiv.HTTPError as exc:
            logger.warning("Skipping curated arXiv id %s after API error: %s", arxiv_id, exc)
            match = None
        except Exception as exc:
            logger.warning("Skipping curated arXiv id %s after unexpected error: %s", arxiv_id, exc)
            match = None
        yield arxiv_id, match


def iter_curated_results(
    client: arxiv.Client,
    titles: list[str],
) -> Iterable[tuple[str, arxiv.Result | None]]:
    for title in titles:
        query = f'ti:"{title}"'
        search = build_search(
            query,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending,
        )
        match = None
        normalized_title = title.casefold()
        try:
            for result in client.results(search):
                candidate = result.title.casefold()
                if normalized_title == candidate:
                    match = result
                    break
                if normalized_title in candidate or candidate in normalized_title:
                    match = result
                    break
                if match is None:
                    match = result
        except arxiv.HTTPError as exc:
            logger.warning("Skipping curated title %r after API error: %s", title, exc)
        except Exception as exc:
            logger.warning("Skipping curated title %r after unexpected error: %s", title, exc)
        yield title, match


def process_result(
    result: arxiv.Result,
    *,
    seen_ids: set[str],
    raw_dir: Path,
    parsed_dir: Path,
    sink,
    min_tokens_per_paper: int,
    request_delay_seconds: float,
) -> tuple[bool, int]:
    paper_id = safe_id(result.get_short_id())
    if paper_id in seen_ids:
        return False, 0

    pdf_path = raw_dir / f"{paper_id}.pdf"
    text_path = parsed_dir / f"{paper_id}.txt"
    text, token_count = extract_text_from_pdf(
        result=result,
        pdf_path=pdf_path,
        min_tokens_per_paper=min_tokens_per_paper,
        request_delay_seconds=request_delay_seconds,
    )
    if token_count < min_tokens_per_paper:
        logger.warning("Skipping %s because it only has %s tokens", paper_id, token_count)
        remove_if_exists(text_path)
        return False, 0

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
    return True, token_count


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

    client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)

    with records_path.open("a", encoding="utf-8") as sink:
        curated_added = 0
        for arxiv_id, result in tqdm(
            iter_curated_id_results(client, settings.dataset.curated_ids),
            total=len(settings.dataset.curated_ids),
            desc="Curated seed ids",
        ):
            if total_tokens >= settings.dataset.target_tokens:
                break
            if result is None:
                logger.warning("No arXiv match found for curated id: %s", arxiv_id)
                continue
            try:
                added, token_count = process_result(
                    result,
                    seen_ids=seen_ids,
                    raw_dir=raw_dir,
                    parsed_dir=parsed_dir,
                    sink=sink,
                    min_tokens_per_paper=settings.dataset.min_tokens_per_paper,
                    request_delay_seconds=settings.dataset.request_delay_seconds,
                )
                if added:
                    curated_added += 1
                    total_tokens += token_count
            except Exception as exc:
                logger.exception("Failed to process curated arXiv id %s: %s", arxiv_id, exc)

        for title, result in tqdm(
            iter_curated_results(client, settings.dataset.curated_titles),
            total=len(settings.dataset.curated_titles),
            desc="Curated seed papers",
        ):
            if total_tokens >= settings.dataset.target_tokens:
                break
            if result is None:
                logger.warning("No arXiv match found for curated paper: %s", title)
                continue
            try:
                added, token_count = process_result(
                    result,
                    seen_ids=seen_ids,
                    raw_dir=raw_dir,
                    parsed_dir=parsed_dir,
                    sink=sink,
                    min_tokens_per_paper=settings.dataset.min_tokens_per_paper,
                    request_delay_seconds=settings.dataset.request_delay_seconds,
                )
                if added:
                    curated_added += 1
                    total_tokens += token_count
            except Exception as exc:
                logger.exception("Failed to process curated paper %s: %s", title, exc)

        search = build_search(
            settings.dataset.query,
            max_results=settings.dataset.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        for result in tqdm(client.results(search), total=settings.dataset.max_results, desc="arXiv papers"):
            if total_tokens >= settings.dataset.target_tokens:
                logger.info("Reached target token count: %s", total_tokens)
                break
            try:
                added, token_count = process_result(
                    result,
                    seen_ids=seen_ids,
                    raw_dir=raw_dir,
                    parsed_dir=parsed_dir,
                    sink=sink,
                    min_tokens_per_paper=settings.dataset.min_tokens_per_paper,
                    request_delay_seconds=settings.dataset.request_delay_seconds,
                )
                if added:
                    total_tokens += token_count
            except Exception as exc:
                logger.exception("Failed to process %s: %s", result.get_short_id(), exc)

    summary = {
        "num_papers": len(seen_ids),
        "total_tokens": total_tokens,
        "query": settings.dataset.query,
        "target_tokens": settings.dataset.target_tokens,
        "curated_ids": settings.dataset.curated_ids,
        "curated_titles": settings.dataset.curated_titles,
    }
    (metadata_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Dataset summary: %s", summary)


if __name__ == "__main__":
    main()
