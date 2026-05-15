#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import time
from pathlib import Path

import arxiv
import fitz
from tqdm import tqdm

try:
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover
    from langchain_community.vectorstores import Chroma

try:
    from langchain_community.document_loaders import PyMuPDFLoader
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install LangChain loaders with `pip install -e .` before building Basic RAG.") from exc

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover
    from langchain.schema import Document

try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:  # pragma: no cover
    from langchain_community.embeddings import OllamaEmbeddings

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.storage.rag_store import BasicRAGStore
from app.utils.tokens import count_tokens

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("build_basic_rag")


def chroma_metadata(metadata: dict) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if isinstance(value, str | int | float | bool):
            clean[key] = value
        elif value is not None:
            clean[key] = str(value)
    return clean


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


def bootstrap_arxiv_corpus(settings, max_results: int) -> int:
    raw_dir = Path(settings.paths.raw_pdfs_dir)
    parsed_dir = Path(settings.paths.parsed_text_dir)
    jsonl_dir = Path(settings.paths.jsonl_dir)
    metadata_dir = Path(settings.paths.metadata_dir)
    for directory in (raw_dir, parsed_dir, jsonl_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    records_path = jsonl_dir / "papers.jsonl"
    seen_ids: set[str] = set()
    if records_path.exists():
        with records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    seen_ids.add(str(json.loads(line).get("id")))

    search = arxiv.Search(
        query=settings.dataset.query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    client = arxiv.Client(page_size=min(max_results, 25), delay_seconds=3, num_retries=3)
    added = 0

    with records_path.open("a", encoding="utf-8") as sink:
        for result in tqdm(client.results(search), total=max_results, desc="Bootstrap arXiv corpus"):
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
                added += 1
            except Exception as exc:
                logger.exception("Failed to bootstrap %s: %s", paper_id, exc)

    summary = {
        "bootstrap_added_papers": added,
        "total_jsonl_records": len(seen_ids),
        "query": settings.dataset.query,
        "max_results": max_results,
    }
    (metadata_dir / "basic_rag_bootstrap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return added


def add_text_document(
    documents: list[Document],
    *,
    path: Path,
    content: str,
    page: int = -1,
    extra_metadata: dict[str, str | int | float | bool] | None = None,
) -> None:
    text = content.strip()
    if not text:
        logger.warning("Skipping empty parsed document: %s", path)
        return
    metadata: dict[str, str | int | float | bool] = {
        "source": str(path),
        "paper_filename": path.name,
        "page": page,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    documents.append(Document(page_content=text, metadata=chroma_metadata(metadata)))


def load_documents(settings) -> list[Document]:
    documents: list[Document] = []
    loaded_stems: set[str] = set()

    markdown_dir = Path(settings.paths.parsed_markdown_dir)
    for markdown_path in sorted(markdown_dir.glob("**/*.md")):
        if markdown_path.stem in loaded_stems:
            continue
        add_text_document(
            documents,
            path=markdown_path,
            content=markdown_path.read_text(encoding="utf-8"),
            extra_metadata={"loader": "parsed_markdown"},
        )
        loaded_stems.add(markdown_path.stem)

    text_dir = Path(settings.paths.parsed_text_dir)
    for text_path in sorted(text_dir.glob("**/*.txt")):
        if text_path.stem in loaded_stems:
            continue
        add_text_document(
            documents,
            path=text_path,
            content=text_path.read_text(encoding="utf-8"),
            extra_metadata={"loader": "parsed_text"},
        )
        loaded_stems.add(text_path.stem)

    records_path = Path(settings.paths.jsonl_dir) / "papers.jsonl"
    if records_path.exists():
        skipped_jsonl = 0
        for line_number, line in enumerate(records_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                skipped_jsonl += 1
                logger.warning("Skipping malformed JSONL record %s:%s: %s", records_path, line_number, exc)
                continue
            text = str(record.get("text") or "").strip()
            if not text:
                continue
            paper_id = str(record.get("id") or f"jsonl-{line_number}")
            if paper_id in loaded_stems:
                continue
            title = str(record.get("title") or paper_id)
            add_text_document(
                documents,
                path=records_path,
                content=text,
                extra_metadata={
                    "loader": "jsonl_record",
                    "paper_filename": f"{paper_id}.jsonl",
                    "paper_id": paper_id,
                    "title": title,
                },
            )
            loaded_stems.add(paper_id)
        if skipped_jsonl:
            logger.warning("Skipped %s malformed JSONL records from %s", skipped_jsonl, records_path)

    pdf_dir = Path(settings.paths.raw_pdfs_dir)
    skipped_pdfs = 0
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        if pdf_path.stem in loaded_stems:
            continue
        try:
            for document in PyMuPDFLoader(str(pdf_path)).load():
                metadata = dict(document.metadata)
                metadata["source"] = str(pdf_path)
                metadata["paper_filename"] = pdf_path.name
                metadata["page"] = int(metadata.get("page", -1))
                metadata["loader"] = "pymupdf_pdf"
                documents.append(Document(page_content=document.page_content, metadata=chroma_metadata(metadata)))
            loaded_stems.add(pdf_path.stem)
        except Exception as exc:
            skipped_pdfs += 1
            logger.warning("Skipping unreadable PDF %s: %s", pdf_path.name, exc)
    if skipped_pdfs:
        logger.warning("Skipped %s unreadable PDFs while building Basic RAG", skipped_pdfs)

    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the persistent Chroma index for PaperWeave Basic RAG.")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--append", action="store_true", help="Append to the existing Chroma collection instead of rebuilding it.")
    parser.add_argument(
        "--bootstrap-arxiv",
        action="store_true",
        help="If no local corpus is present, download a small arXiv corpus before indexing.",
    )
    parser.add_argument("--bootstrap-max-results", type=int, default=8)
    args = parser.parse_args()

    settings = get_settings()
    persist_dir = Path(settings.paths.basic_rag_dir)
    documents = load_documents(settings)
    if not documents and args.bootstrap_arxiv:
        print(f"No local corpus found. Bootstrapping up to {args.bootstrap_max_results} arXiv papers...")
        added = bootstrap_arxiv_corpus(settings, args.bootstrap_max_results)
        print(f"Bootstrapped papers: {added}")
        documents = load_documents(settings)

    print(f"Loaded source documents: {len(documents)}")
    if not documents:
        raise SystemExit(
            "No PDFs, markdown, text, or data/jsonl/papers.jsonl records were found. "
            "Add corpus files before building Basic RAG, or run "
            "`python scripts/build_basic_rag.py --bootstrap-arxiv` to create a small public arXiv corpus."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        source_name = Path(str(chunk.metadata.get("source", "unknown"))).name
        page = chunk.metadata.get("page", -1)
        chunk.metadata["chunk_id"] = f"{source_name}:p{page}:c{index}"
        chunk.metadata["chunk_index"] = index

    print(f"Created chunks: {len(chunks)}")
    if not args.append and persist_dir.exists():
        shutil.rmtree(persist_dir)

    embeddings = OllamaEmbeddings(
        model=settings.providers.embeddings.model,
        base_url=settings.providers.embeddings.base_url or "http://localhost:11434",
    )
    vectorstore = Chroma(
        collection_name=BasicRAGStore.collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    ids = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
    for start in tqdm(range(0, len(chunks), 64), desc="Embedding chunks"):
        end = start + 64
        vectorstore.add_documents(chunks[start:end], ids=ids[start:end])

    if hasattr(vectorstore, "persist"):
        vectorstore.persist()

    stored = int(vectorstore._collection.count())
    summary = {
        "source_documents": len(documents),
        "chunks_created": len(chunks),
        "embeddings_stored": stored,
        "embedding_model": settings.providers.embeddings.model,
        "chroma_dir": str(persist_dir),
        "collection": BasicRAGStore.collection_name,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
    }
    persist_dir.mkdir(parents=True, exist_ok=True)
    (persist_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Total documents loaded: {len(documents)}")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Total embeddings stored: {stored}")
    print(f"Chroma directory: {persist_dir}")


if __name__ == "__main__":
    main()
