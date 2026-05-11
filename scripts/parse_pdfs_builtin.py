#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import fitz
from tqdm import tqdm

from app.core.config import get_settings


def main() -> None:
    settings = get_settings()
    raw_dir = Path(settings.paths.raw_pdfs_dir)
    parsed_dir = Path(settings.paths.parsed_text_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for pdf_path in tqdm(sorted(raw_dir.glob("*.pdf")), desc="Parse PDFs"):
        document = fitz.open(pdf_path)
        text = "\n".join(page.get_text("text") for page in document).strip()
        out_path = parsed_dir / f"{pdf_path.stem}.txt"
        out_path.write_text(text, encoding="utf-8")
        manifest.append({"source_pdf": str(pdf_path), "parsed_text": str(out_path)})
    (Path(settings.paths.metadata_dir) / "parse_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
