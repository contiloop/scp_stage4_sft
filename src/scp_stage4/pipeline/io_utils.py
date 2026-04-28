"""Small JSONL I/O helpers for local contract checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8", newline="\n") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if line.strip() == "":
                raise ValueError(f"{path_obj}:{line_no} contains an empty line")

            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path_obj}:{line_no} must contain a JSON object")
            rows.append(value)
    return rows


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8", newline="\n") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if line.strip() == "":
                raise ValueError(f"{path_obj}:{line_no} contains an empty line")

            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path_obj}:{line_no} must contain a JSON object")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
