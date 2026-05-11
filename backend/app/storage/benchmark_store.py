from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import Settings


class BenchmarkStore:
    def __init__(self, settings: Settings):
        self.output_dir = Path(settings.paths.benchmark_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, payload: dict[str, Any]) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.output_dir / f"benchmark_{timestamp}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
