from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .jsonl import read_jsonl


class RowIdValidationError(ValueError):
    """Raised when row ids drift between artifacts."""


def extract_row_ids(rows: Sequence[Mapping[str, Any]], *, row_id_key: str = "id") -> list[str]:
    ids: list[str] = []
    for index, row in enumerate(rows):
        value = row.get(row_id_key)
        if not isinstance(value, str) or value.strip() == "":
            raise RowIdValidationError(
                f"row index {index} missing non-empty string row id in key '{row_id_key}'"
            )
        ids.append(value)
    return ids


def validate_row_id_preservation(
    base_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    allow_subset: bool = False,
    base_name: str = "base",
    candidate_name: str = "candidate",
    row_id_key: str = "id",
) -> None:
    base_ids = extract_row_ids(base_rows, row_id_key=row_id_key)
    candidate_ids = extract_row_ids(candidate_rows, row_id_key=row_id_key)

    if len(set(base_ids)) != len(base_ids):
        raise RowIdValidationError(f"{base_name} has duplicate row ids")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RowIdValidationError(f"{candidate_name} has duplicate row ids")

    if not allow_subset:
        if len(base_ids) != len(candidate_ids):
            raise RowIdValidationError(
                f"{candidate_name} row count {len(candidate_ids)} does not match {base_name} row count {len(base_ids)}"
            )

        for idx, (left, right) in enumerate(zip(base_ids, candidate_ids)):
            if left != right:
                raise RowIdValidationError(
                    f"row id drift at index {idx}: {base_name}='{left}' vs {candidate_name}='{right}'"
                )
        return

    base_order = {row_id: idx for idx, row_id in enumerate(base_ids)}
    previous = -1
    for idx, row_id in enumerate(candidate_ids):
        if row_id not in base_order:
            raise RowIdValidationError(
                f"{candidate_name} row id '{row_id}' at index {idx} not found in {base_name}"
            )
        current = base_order[row_id]
        if current <= previous:
            raise RowIdValidationError(
                f"{candidate_name} row ids are not in {base_name} order at index {idx}"
            )
        previous = current


def validate_row_id_preservation_files(
    base_path: str | Path,
    candidate_path: str | Path,
    *,
    allow_subset: bool = False,
    base_name: str = "base",
    candidate_name: str = "candidate",
    row_id_key: str = "id",
) -> None:
    base_rows = read_jsonl(base_path)
    candidate_rows = read_jsonl(candidate_path)
    validate_row_id_preservation(
        base_rows,
        candidate_rows,
        allow_subset=allow_subset,
        base_name=base_name,
        candidate_name=candidate_name,
        row_id_key=row_id_key,
    )

