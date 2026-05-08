import json
from pathlib import Path
from typing import Any

from src.config import LOG_DIR
from src.utils import ensure_directories, iso_timestamp


def _default_log_path() -> Path:
    return LOG_DIR / "events.jsonl"


def write_event(event_type: str, payload: dict[str, Any], *, severity: str = "INFO", path: Path | None = None) -> None:
    log_path = path or _default_log_path()
    ensure_directories(log_path.parent)
    record = {
        "timestamp": iso_timestamp(),
        "severity": severity,
        "event_type": event_type,
        "payload": payload,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def read_recent_events(limit: int = 200, *, path: Path | None = None) -> list[dict[str, Any]]:
    log_path = path or _default_log_path()
    if not log_path.exists():
        return []
    with log_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()[-limit:]
    records: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records
