from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, TypeVar

T = TypeVar("T")


class JSONLFormatError(ValueError):
    """Raised when a JSONL file is malformed or cannot be safely decoded."""


def _coerce_row_mapping(row: Any) -> dict[str, Any]:
    if hasattr(row, "to_dict") and callable(row.to_dict):
        row = row.to_dict()
    elif is_dataclass(row):
        row = asdict(row)

    if not isinstance(row, Mapping):
        raise JSONLFormatError(f"row must be a mapping/object, got {type(row)!r}")

    return dict(row)


def read_jsonl(
    path: str | Path,
    *,
    validator: Callable[[Mapping[str, Any]], T] | None = None,
) -> list[T | dict[str, Any]]:
    rows: list[T | dict[str, Any]] = []
    path_obj = Path(path)

    with path_obj.open("r", encoding="utf-8", newline="\n") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if line.strip() == "":
                raise JSONLFormatError(f"{path_obj}:{line_no} contains an empty line")

            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JSONLFormatError(f"{path_obj}:{line_no} invalid JSON: {exc.msg}") from exc

            if not isinstance(parsed, Mapping):
                raise JSONLFormatError(f"{path_obj}:{line_no} must contain a JSON object")

            if validator is None:
                rows.append(dict(parsed))
            else:
                rows.append(validator(parsed))

    return rows


def write_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any] | Any],
    *,
    ensure_ascii: bool = False,
    sort_keys: bool = True,
) -> int:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with path_obj.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            payload = _coerce_row_mapping(row)
            serialized = json.dumps(
                payload,
                ensure_ascii=ensure_ascii,
                sort_keys=sort_keys,
                separators=(",", ":"),
            )
            handle.write(serialized)
            handle.write("\n")
            count += 1

    return count

