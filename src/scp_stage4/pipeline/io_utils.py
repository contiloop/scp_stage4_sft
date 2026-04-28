"""Small JSONL I/O helpers for local contract checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        value = json.loads(stripped)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row must be an object: {path}")
        rows.append(value)
    return rows


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        value = json.loads(stripped)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row must be an object: {path}")
        yield value


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
