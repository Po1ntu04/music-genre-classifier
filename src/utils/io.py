from __future__ import annotations

import csv
import json
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv_rows(path: str | Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
