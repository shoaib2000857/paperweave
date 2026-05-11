#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
from pathlib import Path

from tqdm import tqdm

from app.core.config import get_settings


def main() -> None:
    settings = get_settings()
    raw_dir = Path(settings.paths.raw_pdfs_dir)
    markdown_dir = Path(settings.paths.parsed_markdown_dir)
    markdown_dir.mkdir(parents=True, exist_ok=True)

    module = importlib.import_module("opendataloader")
    convert = getattr(module, "convert")
    manifest = []
    for pdf_path in tqdm(sorted(raw_dir.glob("*.pdf")), desc="OpenDataLoader parse"):
        result = convert(str(pdf_path), output_format="markdown")
        content = result if isinstance(result, str) else str(result)
        out_path = markdown_dir / f"{pdf_path.stem}.md"
        out_path.write_text(content, encoding="utf-8")
        manifest.append({"source_pdf": str(pdf_path), "markdown_path": str(out_path)})
    (Path(settings.paths.metadata_dir) / "opendataloader_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
