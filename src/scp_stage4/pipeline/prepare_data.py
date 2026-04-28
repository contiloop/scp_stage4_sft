"""Local prepare-data implementation for contract harness."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

from scp_stage4.config.loader import compose_config
from scp_stage4.config.validator import validate_config
from scp_stage4.data import read_jsonl, write_jsonl
from scp_stage4.schema import validate_artifact_rows


class PrepareDataError(RuntimeError):
    """Raised when prepare-data contract cannot be satisfied."""


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _get_by_dotpath(data: Mapping[str, Any], key: str) -> Any:
    cursor: Any = data
    for part in key.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _first_non_empty(values: Iterable[Any]) -> str | None:
    for value in values:
        if isinstance(value, str):
            normalized = _normalize_whitespace(value)
            if normalized:
                return normalized
    return None


def _infer_document_type(dataset_name: str, row: Mapping[str, Any]) -> str | None:
    existing = _get_by_dotpath(row, "metadata.document_type")
    if isinstance(existing, str) and existing in {"article", "filing", "earnings_call", "other"}:
        return existing

    lowered = dataset_name.lower()
    if "reuter" in lowered or "bloomberg" in lowered:
        return "article"
    if "10k" in lowered or "sec" in lowered or "filing" in lowered:
        return "filing"
    if "earnings" in lowered or "call" in lowered:
        return "earnings_call"
    return "other"


def _fallback_raw_rows(dataset_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(128):
        rows.append(
            {
                "id": f"{dataset_name}:{idx:06d}",
                "dataset": dataset_name,
                "source_text": (
                    f"{dataset_name} source sentence {idx} with deterministic fixture text "
                    f"for local prepare-data contract checks."
                ),
                "title": f"{dataset_name} title {idx}",
            }
        )
    return rows


def _load_raw_rows(data_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        Path("tests/fixtures/raw.train.jsonl"),
        Path("tests/fixtures/input.happy.jsonl"),
    ]
    for path in candidates:
        if path.exists():
            rows = read_jsonl(path)
            return [dict(row) for row in rows]

    dataset_name = "local_fixture_dataset"
    datasets = data_cfg.get("datasets")
    if isinstance(datasets, list) and datasets:
        first = datasets[0]
        if isinstance(first, Mapping):
            dataset_name = str(first.get("name", dataset_name))
    return _fallback_raw_rows(dataset_name)


def _normalize_rows(raw_rows: list[dict[str, Any]], data_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    dataset_default = "local_fixture_dataset"
    datasets = data_cfg.get("datasets")
    if isinstance(datasets, list) and datasets:
        first = datasets[0]
        if isinstance(first, Mapping):
            dataset_default = str(first.get("name", dataset_default))

    source_columns = data_cfg.get("text_columns", ["source_text", "text", "source"])
    if not isinstance(source_columns, list) or not source_columns:
        source_columns = ["source_text", "text", "source"]
    else:
        source_columns = [str(col) for col in source_columns]
        if "source" not in source_columns:
            # Accept already-normalized fixture rows in local harness.
            source_columns.append("source")

    title_candidates = [
        "metadata.Headline",
        "metadata.headline",
        "metadata.title",
        "Headline",
        "headline",
        "title",
    ]

    translatable_fields = data_cfg.get("translatable_fields")
    normalized: list[dict[str, Any]] = []

    for index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            continue
        dataset_name = str(row.get("dataset", dataset_default))
        base_id = str(row.get("id", f"{dataset_name}:{index:06d}"))
        source = _first_non_empty(_get_by_dotpath(row, key) for key in source_columns)
        title = _first_non_empty(_get_by_dotpath(row, key) for key in title_candidates)
        document_type = _infer_document_type(dataset_name, row)

        emitted = False
        if isinstance(translatable_fields, list) and translatable_fields:
            for field in translatable_fields:
                if not isinstance(field, Mapping):
                    continue
                field_name = str(field.get("name", "field"))
                columns = field.get("columns", [])
                if not isinstance(columns, list) or not columns:
                    continue
                value = _first_non_empty(_get_by_dotpath(row, col) for col in columns)
                if value is None:
                    optional = bool(field.get("optional", False))
                    if optional:
                        continue
                    continue
                text_role = str(field.get("text_role", "other"))
                normalized.append(
                    {
                        "id": f"{base_id}__{field_name}",
                        "dataset": dataset_name,
                        "source": value,
                        "metadata": {
                            "title": title,
                            "document_type": document_type,
                            "text_role": text_role,
                            "original_id": base_id,
                            "parent_id": None,
                            "chunk_idx": None,
                        },
                    }
                )
                emitted = True

        if emitted:
            continue
        if source is None:
            continue

        normalized.append(
            {
                "id": base_id,
                "dataset": dataset_name,
                "source": source,
                "metadata": {
                    "title": title,
                    "document_type": document_type,
                    "text_role": "body",
                    "original_id": base_id,
                    "parent_id": None,
                    "chunk_idx": None,
                },
            }
        )

    return validate_artifact_rows(normalized, "normalized")


def _estimate_tokens(text: str) -> int:
    return len(text.split())


def _apply_length_policy(rows: list[dict[str, Any]], cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    length_cfg = cfg.get("length", {})
    if not isinstance(length_cfg, Mapping) or not bool(length_cfg.get("enabled", True)):
        return rows

    max_source_tokens = int(length_cfg.get("max_source_tokens", 4000))
    overflow = str(length_cfg.get("overflow", "split"))
    split_cfg = length_cfg.get("split", {})
    if not isinstance(split_cfg, Mapping):
        split_cfg = {}
    max_chunks = int(split_cfg.get("max_chunks_per_row", 4))

    filtered: list[dict[str, Any]] = []
    for row in rows:
        source = str(row["source"])
        tokens = _estimate_tokens(source)
        if tokens <= max_source_tokens:
            filtered.append(row)
            continue

        if overflow == "skip":
            continue

        if overflow == "truncate":
            words = source.split()
            truncated = " ".join(words[:max_source_tokens])
            out = dict(row)
            out["source"] = truncated
            filtered.append(out)
            continue

        # overflow == split
        words = source.split()
        chunk_size = max(1, max_source_tokens)
        for chunk_idx, start in enumerate(range(0, len(words), chunk_size)):
            if chunk_idx >= max_chunks:
                break
            chunk_words = words[start : start + chunk_size]
            if not chunk_words:
                continue
            out = dict(row)
            out["id"] = f"{row['id']}__chunk_{chunk_idx}"
            out["source"] = " ".join(chunk_words)
            metadata = dict(row.get("metadata", {}))
            metadata["parent_id"] = row["id"]
            metadata["chunk_idx"] = chunk_idx
            out["metadata"] = metadata
            filtered.append(out)

    return validate_artifact_rows(filtered, "normalized")


def _split_train_eval(
    rows: list[dict[str, Any]], eval_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    total = len(rows)
    if total == 0:
        return [], []
    if total == 1 or eval_ratio <= 0:
        return list(rows), []

    eval_count = int(math.ceil(total * eval_ratio))
    eval_count = max(1, min(eval_count, total - 1))

    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    eval_set = set(indices[:eval_count])

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if idx in eval_set:
            eval_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, eval_rows


def _sample_train_rows(
    train_rows: list[dict[str, Any]],
    subset_size: int | None,
    strategy: str,
    seed: int,
) -> list[dict[str, Any]]:
    if subset_size is None:
        return list(train_rows)
    if not train_rows:
        return []

    size = max(1, min(int(subset_size), len(train_rows)))
    if strategy == "random":
        indices = list(range(len(train_rows)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        selected = sorted(indices[:size])
        return [train_rows[i] for i in selected]
    # first_n default
    return list(train_rows[:size])


def _write_ood_placeholder(data_cfg: Mapping[str, Any], out_path: Path) -> None:
    ood_cfg = data_cfg.get("ood_test", {})
    if not isinstance(ood_cfg, Mapping) or not bool(ood_cfg.get("enabled", False)):
        return

    source_path = Path(str(ood_cfg.get("path", "")))
    if not source_path.exists():
        write_jsonl(out_path, [])
        return

    source_col = str(ood_cfg.get("source_column", "Source_En"))
    target_col = str(ood_cfg.get("target_column", "Target_Ko"))
    rows: list[dict[str, Any]] = []
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            source = _normalize_whitespace(str(row.get(source_col, "")))
            target = _normalize_whitespace(str(row.get(target_col, "")))
            if not source:
                continue
            rows.append(
                {
                    "id": f"ood_{idx:06d}",
                    "source": source,
                    "target": target,
                }
            )
    write_jsonl(out_path, rows)


def run_prepare_data(
    config_path: str = "configs/scp_stage4.yaml",
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    cfg = compose_config(config_path, overrides=overrides)
    validate_config(cfg)

    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, Mapping):
        raise PrepareDataError("data config must be a mapping")

    raw_rows = _load_raw_rows(data_cfg)
    normalized = _normalize_rows(raw_rows, data_cfg)
    normalized = _apply_length_policy(normalized, data_cfg)

    split_cfg = data_cfg.get("split", {})
    if not isinstance(split_cfg, Mapping):
        split_cfg = {}
    eval_ratio = float(split_cfg.get("eval_ratio", 0.02))
    split_seed = int(split_cfg.get("seed", 42))
    train_rows, eval_rows = _split_train_eval(normalized, eval_ratio, split_seed)

    sampling_cfg = data_cfg.get("sampling", {})
    if not isinstance(sampling_cfg, Mapping):
        sampling_cfg = {}
    sampling_strategy = str(sampling_cfg.get("strategy", "first_n"))
    sampling_seed = int(sampling_cfg.get("seed", 42))
    subset_size = data_cfg.get("subset_size")
    sampled_rows = _sample_train_rows(
        train_rows,
        int(subset_size) if subset_size is not None else None,
        sampling_strategy,
        sampling_seed,
    )

    out_dir = Path("artifacts/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "datapool.normalized.jsonl", normalized)
    write_jsonl(out_dir / "datapool.train.jsonl", train_rows)
    write_jsonl(out_dir / "datapool.eval.jsonl", eval_rows)
    write_jsonl(out_dir / "datapool.train.sampled.jsonl", sampled_rows)
    _write_ood_placeholder(data_cfg, out_dir / "ood_test.jsonl")

    summary = {
        "normalized_rows": len(normalized),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "sampled_rows": len(sampled_rows),
        "artifact_dir": str(out_dir),
    }
    (out_dir / "prepare_data_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare local data artifacts")
    parser.add_argument("--config", default="configs/scp_stage4.yaml")
    args, overrides = parser.parse_known_args(argv)
    summary = run_prepare_data(config_path=args.config, overrides=overrides)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
