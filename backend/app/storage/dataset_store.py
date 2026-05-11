from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.config import Settings


class DatasetStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.paper_path = Path(settings.paths.jsonl_dir) / "papers.jsonl"

    def load_papers(self) -> list[dict[str, Any]]:
        if not self.paper_path.exists():
            return []
        with self.paper_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
