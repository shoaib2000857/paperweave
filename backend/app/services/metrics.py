from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import Settings


class MetricsService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.path = Path(settings.app.metrics_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, payload: dict[str, Any]) -> None:
        metrics = self.read_metrics()
        metrics.setdefault("events", []).append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **payload,
            }
        )
        self.path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def read_metrics(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"events": []}
        return json.loads(self.path.read_text(encoding="utf-8"))
