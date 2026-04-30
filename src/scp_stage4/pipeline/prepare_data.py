"""Local prepare-data implementation for contract harness."""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import math
import random
import re
import shutil
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from scp_stage4.config.loader import compose_config
from scp_stage4.config.validator import validate_config
from scp_stage4.data import read_jsonl, write_jsonl
from scp_stage4.schema import validate_artifact_row, validate_artifact_rows

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]


class PrepareDataError(RuntimeError):
    """Raised when prepare-data contract cannot be satisfied."""


class _JsonlStreamWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8", newline="\n")
        self.count = 0

    def write(self, row: Mapping[str, Any]) -> None:
        self._handle.write(
            json.dumps(
                dict(row),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        self._handle.write("\n")
        self.count += 1

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "_JsonlStreamWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, exc_tb: object) -> None:
        self.close()


class _ProgressReporter:
    def __init__(
        self,
        *,
        phase: str,
        enabled: bool,
        every_rows: int,
        every_seconds: float,
    ) -> None:
        self.phase = phase
        self.enabled = enabled
        self.every_rows = max(1, int(every_rows))
        self.every_seconds = max(0.1, float(every_seconds))
        self._start = time.perf_counter()
        self._last_report_time = self._start
        self._last_report_rows = 0

    def maybe_report(self, rows: int) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        row_delta = rows - self._last_report_rows
        time_delta = now - self._last_report_time
        if row_delta < self.every_rows and time_delta < self.every_seconds:
            return
        self._emit(rows=rows, now=now)

    def finish(self, rows: int) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        if rows != self._last_report_rows:
            self._emit(rows=rows, now=now, done=True)
        else:
            total_elapsed = max(now - self._start, 1e-9)
            total_rps = rows / total_elapsed
            print(
                (
                    f"[prepare-data] phase={self.phase} rows={rows} "
                    f"rows_per_sec=0.0 avg_rows_per_sec={total_rps:.1f} "
                    f"elapsed_sec={total_elapsed:.1f} done=true"
                ),
                file=sys.stderr,
                flush=True,
            )

    def _emit(self, *, rows: int, now: float, done: bool = False) -> None:
        row_delta = max(rows - self._last_report_rows, 0)
        time_delta = max(now - self._last_report_time, 1e-9)
        total_elapsed = max(now - self._start, 1e-9)
        window_rps = row_delta / time_delta
        total_rps = rows / total_elapsed
        done_text = "true" if done else "false"
        print(
            (
                f"[prepare-data] phase={self.phase} rows={rows} "
                f"rows_per_sec={window_rps:.1f} avg_rows_per_sec={total_rps:.1f} "
                f"elapsed_sec={total_elapsed:.1f} done={done_text}"
            ),
            file=sys.stderr,
            flush=True,
        )
        self._last_report_time = now
        self._last_report_rows = rows


def _normalized_parquet_schema() -> Any:
    if pa is None:
        return None
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("dataset", pa.string()),
            pa.field("source", pa.string()),
            pa.field(
                "metadata",
                pa.struct(
                    [
                        pa.field("title", pa.string()),
                        pa.field("document_type", pa.string()),
                        pa.field("text_role", pa.string()),
                        pa.field("original_id", pa.string()),
                        pa.field("parent_id", pa.string()),
                        pa.field("chunk_idx", pa.int64()),
                    ]
                ),
            ),
        ]
    )


class _ParquetStreamWriter:
    def __init__(self, path: Path, *, row_group_size: int = 4096) -> None:
        if pa is None or pq is None:
            raise PrepareDataError(
                "Parquet intermediate format requires 'pyarrow' to be installed"
            )
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._row_group_size = max(1, int(row_group_size))
        self._pending: list[dict[str, Any]] = []
        self._schema = _normalized_parquet_schema()
        self._writer: Any | None = None
        self.count = 0

    def _flush_pending(self) -> None:
        if not self._pending:
            return
        table = pa.Table.from_pylist(self._pending, schema=self._schema)
        if self._writer is None:
            self._writer = pq.ParquetWriter(str(self.path), table.schema, compression="zstd")
        self._writer.write_table(table)
        self._pending.clear()

    def write(self, row: Mapping[str, Any]) -> None:
        self._pending.append(dict(row))
        self.count += 1
        if len(self._pending) >= self._row_group_size:
            self._flush_pending()

    def close(self) -> None:
        self._flush_pending()
        if self._writer is None:
            empty_table = pa.Table.from_pylist([], schema=self._schema)
            self._writer = pq.ParquetWriter(
                str(self.path),
                empty_table.schema,
                compression="zstd",
            )
        self._writer.close()

    def __enter__(self) -> "_ParquetStreamWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, exc_tb: object) -> None:
        self.close()


def _iter_parquet_mapping_rows(path: Path, *, batch_size: int = 4096) -> Iterable[dict[str, Any]]:
    if pa is None or pq is None:
        raise PrepareDataError("Parquet reading requires 'pyarrow' to be installed")
    parquet_file = pq.ParquetFile(str(path))
    for record_batch in parquet_file.iter_batches(batch_size=max(1, int(batch_size))):
        table = pa.Table.from_batches([record_batch])
        for row in table.to_pylist():
            if isinstance(row, Mapping):
                yield dict(row)


class _TokenCounter:
    def __init__(
        self,
        *,
        count: Callable[[str], int],
        count_batch: Callable[[list[str]], list[int]],
    ) -> None:
        self._count = count
        self._count_batch = count_batch

    def count(self, text: str) -> int:
        return self._count(text)

    def count_batch(self, texts: list[str]) -> list[int]:
        return self._count_batch(texts)


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
                "source_text": f"Fixture sentence number {idx} for checks.",
                "title": f"{dataset_name} title {idx}",
            }
        )
    return rows


def _fixture_raw_rows(data_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
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


def _iter_local_jsonl_rows(data_cfg: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    runtime_cfg = data_cfg.get("runtime", {})
    if not isinstance(runtime_cfg, Mapping):
        raise PrepareDataError("data.runtime must be a mapping")
    path_value = runtime_cfg.get("local_jsonl_path")
    if not isinstance(path_value, str) or not path_value.strip():
        raise PrepareDataError(
            "data.runtime.local_jsonl_path is required when data.runtime.mode=local_jsonl"
        )
    path = Path(path_value)
    if not path.exists():
        raise PrepareDataError(f"local JSONL dataset not found: {path}")
    yield from _iter_jsonl_mapping_rows(path)


def _iter_jsonl_mapping_rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise PrepareDataError(
                    f"Invalid JSONL record at {path}:{line_idx}: {exc}"
                ) from exc
            if isinstance(payload, Mapping):
                yield dict(payload)


def _append_data_file_patterns(patterns: list[str], value: Any) -> None:
    if isinstance(value, str):
        candidate = value.strip()
        if candidate:
            patterns.append(candidate)
        return

    if isinstance(value, Mapping):
        for nested in value.values():
            _append_data_file_patterns(patterns, nested)
        return

    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        for nested in value:
            _append_data_file_patterns(patterns, nested)


def _resolve_data_file_patterns(spec: Mapping[str, Any], split: str) -> list[str]:
    raw_data_files = spec.get("data_files")
    patterns: list[str] = []
    if isinstance(raw_data_files, Mapping) and split in raw_data_files:
        _append_data_file_patterns(patterns, raw_data_files.get(split))
    elif raw_data_files is not None:
        _append_data_file_patterns(patterns, raw_data_files)

    if not patterns:
        patterns = [
            "data/*.jsonl",
            "*.jsonl",
            "data/*.jsonl.gz",
            "*.jsonl.gz",
            "data/*.ndjson",
            "*.ndjson",
        ]

    deduped: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        if pattern in seen:
            continue
        seen.add(pattern)
        deduped.append(pattern)
    return deduped


def _with_row_contract(raw_row: Mapping[str, Any], dataset_name: str, row_index: int) -> dict[str, Any]:
    row = dict(raw_row)
    row.setdefault("dataset", dataset_name)
    raw_id = row.get("id") or row.get("_id") or row.get("doc_id")
    row["id"] = str(raw_id) if raw_id is not None else f"{dataset_name}:{row_index:08d}"
    return row


def _load_hf_rows_via_snapshot_jsonl(
    dataset_name: str,
    split: str,
    spec: Mapping[str, Any],
    max_rows_per_dataset: int | None,
) -> Iterable[dict[str, Any]]:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ModuleNotFoundError as exc:
        raise PrepareDataError(
            "HF JSONL fallback requires the 'huggingface_hub' package"
        ) from exc

    revision = spec.get("revision")
    revision_value = str(revision) if isinstance(revision, str) and revision.strip() else None
    allow_patterns = _resolve_data_file_patterns(spec, split)
    snapshot_path = Path(
        snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            revision=revision_value,
            allow_patterns=allow_patterns,
        )
    )

    jsonl_paths = sorted(snapshot_path.rglob("*.jsonl"))
    jsonl_paths += sorted(snapshot_path.rglob("*.jsonl.gz"))
    jsonl_paths += sorted(snapshot_path.rglob("*.ndjson"))
    jsonl_paths += sorted(snapshot_path.rglob("*.ndjson.gz"))
    if not jsonl_paths:
        raise PrepareDataError(
            f"HF JSONL fallback found no JSONL files in snapshot: {snapshot_path}"
        )

    row_index = 0
    emitted = False
    for path in jsonl_paths:
        for raw_row in _iter_jsonl_mapping_rows(path):
            if max_rows_per_dataset is not None and row_index >= max_rows_per_dataset:
                break
            emitted = True
            yield _with_row_contract(raw_row, dataset_name, row_index)
            row_index += 1
        if max_rows_per_dataset is not None and row_index >= max_rows_per_dataset:
            break

    if not emitted:
        raise PrepareDataError(
            f"HF JSONL fallback produced zero rows for dataset '{dataset_name}'"
        )


def _load_hf_rows(data_cfg: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ModuleNotFoundError as exc:
        raise PrepareDataError(
            "data.runtime.mode=hf requires the Hugging Face 'datasets' package"
        ) from exc

    runtime_cfg = data_cfg.get("runtime", {})
    if not isinstance(runtime_cfg, Mapping):
        runtime_cfg = {}
    hf_cfg = runtime_cfg.get("hf", {})
    if not isinstance(hf_cfg, Mapping):
        hf_cfg = {}

    streaming = bool(hf_cfg.get("streaming", False))
    fallback_to_snapshot_jsonl = bool(hf_cfg.get("fallback_to_snapshot_jsonl", True))
    max_rows_raw = hf_cfg.get("max_rows_per_dataset")
    max_rows_per_dataset = int(max_rows_raw) if max_rows_raw is not None else None
    raw_dataset_download_workers = hf_cfg.get("dataset_download_workers", 1)
    if isinstance(raw_dataset_download_workers, bool) or not isinstance(
        raw_dataset_download_workers, int
    ):
        dataset_download_workers = 1
    else:
        dataset_download_workers = max(1, raw_dataset_download_workers)
    raw_num_workers = data_cfg.get("num_workers", 1)
    if isinstance(raw_num_workers, bool) or not isinstance(raw_num_workers, int):
        num_workers = 1
    else:
        num_workers = max(1, raw_num_workers)

    dataset_specs = data_cfg.get("datasets")
    if not isinstance(dataset_specs, list) or not dataset_specs:
        raise PrepareDataError("data.datasets must be a non-empty list for HF loading")

    def _load_one_dataset(dataset_index: int, spec: Mapping[str, Any]) -> tuple[str, str, Any]:
        if not isinstance(spec, Mapping):
            raise PrepareDataError(f"data.datasets[{dataset_index}] must be a mapping")
        name = str(spec.get("name", "")).strip()
        if not name:
            raise PrepareDataError(f"data.datasets[{dataset_index}].name is required")
        split = str(spec.get("split", "train"))

        load_kwargs: dict[str, Any] = {
            "split": split,
            "streaming": streaming,
        }
        if not streaming and num_workers > 1:
            load_kwargs["num_proc"] = num_workers
        for optional_key in (
            "data_dir",
            "data_files",
            "revision",
            "trust_remote_code",
        ):
            if optional_key in spec:
                load_kwargs[optional_key] = spec[optional_key]

        config_name = spec.get("config_name")
        try:
            if isinstance(config_name, str) and config_name.strip():
                dataset = load_dataset(name, config_name, **load_kwargs)
            else:
                dataset = load_dataset(name, **load_kwargs)
            return "dataset", name, dataset
        except Exception as exc:
            if not fallback_to_snapshot_jsonl:
                raise PrepareDataError(
                    f"HF load_dataset failed for '{name}' and fallback is disabled: {exc}"
                ) from exc
            return (
                "iterable",
                name,
                _load_hf_rows_via_snapshot_jsonl(
                    dataset_name=name,
                    split=split,
                    spec=spec,
                    max_rows_per_dataset=max_rows_per_dataset,
                ),
            )

    def _iter_dataset_rows(
        kind: str,
        dataset_name: str,
        payload: Any,
    ) -> Iterable[dict[str, Any]]:
        if kind == "iterable":
            yield from payload
            return
        row_index = 0
        for raw_row in payload:
            if max_rows_per_dataset is not None and row_index >= max_rows_per_dataset:
                break
            if not isinstance(raw_row, Mapping):
                continue
            yield _with_row_contract(raw_row, dataset_name, row_index)
            row_index += 1

    def _iter_rows() -> Iterable[dict[str, Any]]:
        emitted = False
        if dataset_download_workers <= 1 or len(dataset_specs) <= 1:
            for dataset_index, spec in enumerate(dataset_specs):
                kind, dataset_name, payload = _load_one_dataset(dataset_index, spec)
                for row in _iter_dataset_rows(kind, dataset_name, payload):
                    emitted = True
                    yield row
        else:
            max_parallel = min(dataset_download_workers, len(dataset_specs))
            indexed_rows: dict[int, tuple[str, str, Any]] = {}
            with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                future_to_index = {
                    executor.submit(_load_one_dataset, dataset_index, spec): dataset_index
                    for dataset_index, spec in enumerate(dataset_specs)
                }
                for future, dataset_index in future_to_index.items():
                    indexed_rows[dataset_index] = future.result()
            for dataset_index in range(len(dataset_specs)):
                prepared = indexed_rows.get(dataset_index)
                if prepared is None:
                    continue
                kind, dataset_name, payload = prepared
                for row in _iter_dataset_rows(kind, dataset_name, payload):
                    emitted = True
                    yield row
        if not emitted:
            raise PrepareDataError("HF loading produced zero rows")

    return _iter_rows()


def _iter_raw_rows(data_cfg: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    runtime_cfg = data_cfg.get("runtime", {})
    if not isinstance(runtime_cfg, Mapping):
        runtime_cfg = {}
    mode = str(runtime_cfg.get("mode", "fixture"))
    if mode == "fixture":
        yield from _fixture_raw_rows(data_cfg)
        return
    if mode == "local_jsonl":
        yield from _iter_local_jsonl_rows(data_cfg)
        return
    if mode == "hf":
        yield from _load_hf_rows(data_cfg)
        return
    raise PrepareDataError(f"Unsupported data.runtime.mode: {mode}")


def _load_raw_rows(data_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(_iter_raw_rows(data_cfg))


def _iter_normalized_rows(
    raw_rows: Iterable[dict[str, Any]],
    data_cfg: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
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
                yield {
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
                emitted = True

        if emitted:
            continue
        if source is None:
            continue

        yield {
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


def _normalize_rows(raw_rows: list[dict[str, Any]], data_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    return validate_artifact_rows(_iter_normalized_rows(raw_rows, data_cfg), "normalized")


def _estimate_tokens(text: str) -> int:
    return len(text.split())


def _build_token_counter(cfg: Mapping[str, Any]) -> _TokenCounter:
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, Mapping):
        data_cfg = {}
    length_cfg = data_cfg.get("length", {})
    if not isinstance(length_cfg, Mapping):
        length_cfg = {}

    mode = str(length_cfg.get("mode", "whitespace"))
    if mode != "tokenizer":
        return _TokenCounter(
            count=_estimate_tokens,
            count_batch=lambda texts: [_estimate_tokens(text) for text in texts],
        )

    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, Mapping):
        model_cfg = {}
    tokenizer_name = length_cfg.get("tokenizer_name") or model_cfg.get("name")
    fallback = str(length_cfg.get("tokenizer_fallback", "whitespace"))
    local_files_only = bool(length_cfg.get("tokenizer_local_files_only", False))
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))

    try:
        from transformers import AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_name),
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        if fallback == "whitespace":
            return _TokenCounter(
                count=_estimate_tokens,
                count_batch=lambda texts: [_estimate_tokens(text) for text in texts],
            )
        raise PrepareDataError(
            "tokenizer length mode requires a loadable Hugging Face tokenizer; "
            f"failed to load {tokenizer_name!r}"
        ) from exc

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    def count_tokens_batch(texts: list[str]) -> list[int]:
        if not texts:
            return []
        try:
            encoded = tokenizer(
                texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                return_length=True,
            )
            lengths = encoded.get("length")
            if isinstance(lengths, list):
                return [int(value) for value in lengths]
        except Exception:
            # Fall back to per-row tokenization when fast length path is unavailable.
            pass
        return [count_tokens(text) for text in texts]

    return _TokenCounter(
        count=count_tokens,
        count_batch=count_tokens_batch,
    )


_SENTENCE_RE = re.compile(r"[^.!?。！？\n]+[.!?。！？]?(?:\s+|$)|[^\n]+(?:\n|$)")


def _sentence_units(text: str) -> list[str]:
    units = [_normalize_whitespace(match.group(0)) for match in _SENTENCE_RE.finditer(text)]
    return [unit for unit in units if unit]


def _split_long_source(
    row: Mapping[str, Any],
    *,
    token_count: Callable[[str], int],
    max_tokens_per_chunk: int,
    max_chunks: int,
    fallback_for_long_sentence: str,
    on_max_chunks_exceeded: str,
) -> list[dict[str, Any]]:
    source = str(row["source"])
    chunks: list[str] = []
    current: list[str] = []

    def flush_current() -> None:
        if current:
            chunks.append(_normalize_whitespace(" ".join(current)))
            current.clear()

    for sentence in _sentence_units(source):
        sentence_tokens = token_count(sentence)
        if sentence_tokens > max_tokens_per_chunk:
            flush_current()
            if fallback_for_long_sentence == "truncate":
                chunks.append(" ".join(sentence.split()[:max_tokens_per_chunk]))
            elif fallback_for_long_sentence == "split":
                words = sentence.split()
                for start in range(0, len(words), max_tokens_per_chunk):
                    chunks.append(" ".join(words[start : start + max_tokens_per_chunk]))
            else:
                return []
            continue

        candidate = _normalize_whitespace(" ".join([*current, sentence]))
        if current and token_count(candidate) > max_tokens_per_chunk:
            flush_current()
        current.append(sentence)
    flush_current()

    chunks = [chunk for chunk in chunks if chunk]
    if len(chunks) > max_chunks:
        if on_max_chunks_exceeded == "error":
            raise PrepareDataError(f"row {row['id']} exceeded max_chunks_per_row={max_chunks}")
        return []

    out_rows: list[dict[str, Any]] = []
    for chunk_idx, chunk in enumerate(chunks):
        out = dict(row)
        out["id"] = f"{row['id']}__chunk_{chunk_idx}"
        out["source"] = chunk
        metadata = dict(row.get("metadata", {}))
        metadata["parent_id"] = row["id"]
        metadata["chunk_idx"] = chunk_idx
        out["metadata"] = metadata
        out_rows.append(out)
    return out_rows


def _resolved_runtime_token_limits(cfg: Mapping[str, Any]) -> tuple[int, int]:
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, Mapping):
        data_cfg = {}
    length_cfg = data_cfg.get("length", {})
    if not isinstance(length_cfg, Mapping):
        length_cfg = {}

    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, Mapping):
        model_cfg = {}

    model_max_length = int(model_cfg.get("max_length", 0))
    model_max_seq_length = int(model_cfg.get("max_seq_length", model_max_length))
    max_total_tokens = int(length_cfg.get("max_total_tokens", model_max_length))
    runtime_max_total = min(model_max_length, model_max_seq_length, max_total_tokens)

    prompt_template_tokens = int(length_cfg.get("prompt_template_tokens", 0))
    safety_margin_tokens = int(length_cfg.get("safety_margin_tokens", 0))
    min_available_output_tokens = int(length_cfg.get("min_available_output_tokens", 0))
    max_source_tokens = int(length_cfg.get("max_source_tokens", 0))

    source_budget_by_context = (
        runtime_max_total
        - prompt_template_tokens
        - safety_margin_tokens
        - min_available_output_tokens
    )
    effective_max_source_tokens = min(max_source_tokens, source_budget_by_context)
    if effective_max_source_tokens <= 0:
        raise PrepareDataError(
            "length policy is unsatisfiable: "
            "max_total_tokens - prompt_template_tokens - safety_margin_tokens - "
            "min_available_output_tokens must be > 0"
        )
    return runtime_max_total, effective_max_source_tokens


def _iter_rows_with_length_policy(
    rows: Iterable[dict[str, Any]],
    cfg: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, Mapping):
        data_cfg = {}
    length_cfg = data_cfg.get("length", {})
    if not isinstance(length_cfg, Mapping) or not bool(length_cfg.get("enabled", True)):
        yield from rows
        return

    runtime_max_total, effective_max_source_tokens = _resolved_runtime_token_limits(cfg)
    prompt_template_tokens = int(length_cfg.get("prompt_template_tokens", 0))
    safety_margin_tokens = int(length_cfg.get("safety_margin_tokens", 0))
    min_available_output_tokens = int(length_cfg.get("min_available_output_tokens", 0))
    overflow = str(length_cfg.get("overflow", "split"))
    split_cfg = length_cfg.get("split", {})
    if not isinstance(split_cfg, Mapping):
        split_cfg = {}
    max_chunks = int(split_cfg.get("max_chunks_per_row", 4))
    max_tokens_per_chunk = int(
        split_cfg.get("max_source_tokens_per_chunk", effective_max_source_tokens)
    )
    max_tokens_per_chunk = min(max_tokens_per_chunk, effective_max_source_tokens)
    fallback_for_long_sentence = str(split_cfg.get("fallback_for_long_sentence", "skip"))
    on_max_chunks_exceeded = str(split_cfg.get("on_max_chunks_exceeded", "skip"))
    raw_batch_size = length_cfg.get("tokenizer_batch_size", 512)
    if isinstance(raw_batch_size, bool) or not isinstance(raw_batch_size, int):
        batch_size = 512
    else:
        batch_size = max(1, raw_batch_size)
    token_counter = _build_token_counter(cfg)

    pending_rows: list[dict[str, Any]] = []
    pending_sources: list[str] = []

    def _flush_pending() -> Iterable[dict[str, Any]]:
        if not pending_rows:
            return ()

        source_token_counts = token_counter.count_batch(pending_sources)
        if len(source_token_counts) != len(pending_rows):
            raise PrepareDataError("tokenizer length counting returned mismatched batch length")

        filtered_rows: list[dict[str, Any]] = []
        for row, source_tokens in zip(pending_rows, source_token_counts):
            source = str(row["source"])
            available_output_budget = (
                runtime_max_total
                - prompt_template_tokens
                - source_tokens
                - safety_margin_tokens
            )
            if (
                source_tokens <= effective_max_source_tokens
                and available_output_budget >= min_available_output_tokens
            ):
                filtered_rows.append(row)
                continue

            if overflow == "skip":
                continue

            if overflow == "truncate":
                words = source.split()
                truncated = " ".join(words[:effective_max_source_tokens])
                out = dict(row)
                out["source"] = truncated
                truncated_tokens = token_counter.count(truncated)
                available_after_truncation = (
                    runtime_max_total
                    - prompt_template_tokens
                    - truncated_tokens
                    - safety_margin_tokens
                )
                if available_after_truncation >= min_available_output_tokens:
                    filtered_rows.append(out)
                continue

            filtered_rows.extend(
                _split_long_source(
                    row,
                    token_count=token_counter.count,
                    max_tokens_per_chunk=max_tokens_per_chunk,
                    max_chunks=max_chunks,
                    fallback_for_long_sentence=fallback_for_long_sentence,
                    on_max_chunks_exceeded=on_max_chunks_exceeded,
                )
            )
        pending_rows.clear()
        pending_sources.clear()
        return filtered_rows

    for row in rows:
        pending_rows.append(row)
        pending_sources.append(str(row["source"]))
        if len(pending_rows) >= batch_size:
            yield from _flush_pending()

    yield from _flush_pending()


def _apply_length_policy(rows: list[dict[str, Any]], cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    return validate_artifact_rows(_iter_rows_with_length_policy(rows, cfg), "normalized")


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


def _target_eval_count(total_rows: int, eval_ratio: float) -> int:
    if total_rows <= 1 or eval_ratio <= 0:
        return 0
    eval_count = int(math.ceil(total_rows * eval_ratio))
    return max(1, min(eval_count, total_rows - 1))


def _select_eval_indices_from_draws(
    draws_path: Path,
    *,
    total_rows: int,
    eval_count: int,
) -> set[int]:
    if eval_count <= 0 or total_rows <= 1:
        return set()

    heap: list[tuple[float, int]] = []
    bytes_per_draw = struct.calcsize("<d")
    observed_rows = 0
    with draws_path.open("rb") as handle:
        while True:
            chunk = handle.read(bytes_per_draw)
            if not chunk:
                break
            if len(chunk) != bytes_per_draw:
                raise PrepareDataError(
                    f"corrupted eval draw file: {draws_path}"
                )
            draw = struct.unpack("<d", chunk)[0]
            if len(heap) < eval_count:
                heapq.heappush(heap, (-draw, observed_rows))
            elif draw < -heap[0][0]:
                heapq.heapreplace(heap, (-draw, observed_rows))
            observed_rows += 1

    if observed_rows != total_rows:
        raise PrepareDataError(
            "eval split draw count mismatch; expected "
            f"{total_rows}, observed {observed_rows}"
        )

    return {row_index for _, row_index in heap}


def _build_split_artifacts(
    *,
    normalized_rows_iter: Iterable[dict[str, Any]],
    normalized_output_path: Path,
    train_path: Path,
    eval_path: Path,
    sampled_path: Path,
    eval_indices: set[int],
    subset_size: int | None,
    sampling_strategy: str,
    sampling_seed: int,
    on_processed_row: Callable[[int], None] | None = None,
) -> tuple[int, int, int]:
    train_rows = 0
    eval_rows = 0
    sampled_rows = 0

    subset_limit = int(subset_size) if subset_size is not None else None
    first_n_writer: _JsonlStreamWriter | None = None
    if subset_limit is not None and sampling_strategy != "random":
        first_n_writer = _JsonlStreamWriter(sampled_path)

    random_reservoir: list[tuple[int, dict[str, Any]]] = []
    rng = random.Random(sampling_seed)

    try:
        with (
            _JsonlStreamWriter(normalized_output_path) as normalized_writer,
            _JsonlStreamWriter(train_path) as train_writer,
            _JsonlStreamWriter(eval_path) as eval_writer,
        ):
            for row_index, row in enumerate(normalized_rows_iter):
                validated = validate_artifact_row(row, "normalized")
                normalized_writer.write(validated)
                if on_processed_row is not None:
                    on_processed_row(row_index + 1)
                if row_index in eval_indices:
                    eval_writer.write(validated)
                    eval_rows += 1
                    continue

                train_writer.write(validated)
                train_row_index = train_rows
                train_rows += 1

                if subset_limit is None:
                    continue

                if sampling_strategy == "random":
                    if len(random_reservoir) < subset_limit:
                        random_reservoir.append((train_row_index, validated))
                    else:
                        replacement_index = rng.randint(0, train_row_index)
                        if replacement_index < subset_limit:
                            random_reservoir[replacement_index] = (train_row_index, validated)
                    continue

                if first_n_writer is not None and sampled_rows < subset_limit:
                    first_n_writer.write(validated)
                    sampled_rows += 1
    finally:
        if first_n_writer is not None:
            first_n_writer.close()

    if subset_limit is None:
        shutil.copyfile(train_path, sampled_path)
        sampled_rows = train_rows
        return train_rows, eval_rows, sampled_rows

    if sampling_strategy == "random":
        sorted_sampled_rows = sorted(random_reservoir, key=lambda item: item[0])
        with _JsonlStreamWriter(sampled_path) as sampled_writer:
            for _, sampled_row in sorted_sampled_rows:
                sampled_writer.write(sampled_row)
        sampled_rows = len(sorted_sampled_rows)

    return train_rows, eval_rows, sampled_rows


def _prepare_data_runtime_cfg(data_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    runtime_cfg = data_cfg.get("runtime", {})
    if not isinstance(runtime_cfg, Mapping):
        runtime_cfg = {}
    prepare_cfg = runtime_cfg.get("prepare_data", {})
    if not isinstance(prepare_cfg, Mapping):
        prepare_cfg = {}
    return prepare_cfg


def _resolve_intermediate_format(data_cfg: Mapping[str, Any]) -> tuple[str, int]:
    prepare_cfg = _prepare_data_runtime_cfg(data_cfg)

    requested = str(prepare_cfg.get("intermediate_format", "parquet")).strip().lower()
    if requested not in {"parquet", "jsonl"}:
        requested = "parquet"

    raw_row_group_size = prepare_cfg.get("parquet_row_group_size", 4096)
    if isinstance(raw_row_group_size, bool) or not isinstance(raw_row_group_size, int):
        row_group_size = 4096
    else:
        row_group_size = max(1, raw_row_group_size)

    if requested == "parquet" and (pa is None or pq is None):
        return "jsonl", row_group_size
    return requested, row_group_size


def _resolve_progress_config(data_cfg: Mapping[str, Any]) -> tuple[bool, int, float]:
    prepare_cfg = _prepare_data_runtime_cfg(data_cfg)
    progress_enabled_raw = prepare_cfg.get("progress_enabled", True)
    progress_enabled = bool(progress_enabled_raw)

    raw_every_rows = prepare_cfg.get("progress_every_rows", 100_000)
    if isinstance(raw_every_rows, bool) or not isinstance(raw_every_rows, int):
        every_rows = 100_000
    else:
        every_rows = max(1, raw_every_rows)

    raw_every_seconds = prepare_cfg.get("progress_every_seconds", 10.0)
    if isinstance(raw_every_seconds, bool) or not isinstance(raw_every_seconds, (int, float)):
        every_seconds = 10.0
    else:
        every_seconds = max(0.1, float(raw_every_seconds))

    return progress_enabled, every_rows, every_seconds


def _iter_intermediate_rows(
    *,
    fmt: str,
    path: Path,
    parquet_batch_size: int,
) -> Iterable[dict[str, Any]]:
    if fmt == "parquet":
        yield from _iter_parquet_mapping_rows(path, batch_size=parquet_batch_size)
        return
    yield from _iter_jsonl_mapping_rows(path)


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

    split_cfg = data_cfg.get("split", {})
    if not isinstance(split_cfg, Mapping):
        split_cfg = {}
    eval_ratio = float(split_cfg.get("eval_ratio", 0.02))
    split_seed = int(split_cfg.get("seed", 42))

    sampling_cfg = data_cfg.get("sampling", {})
    if not isinstance(sampling_cfg, Mapping):
        sampling_cfg = {}
    sampling_strategy = str(sampling_cfg.get("strategy", "first_n"))
    sampling_seed = int(sampling_cfg.get("seed", 42))
    subset_size = data_cfg.get("subset_size")
    subset_limit = int(subset_size) if subset_size is not None else None

    out_dir = Path("artifacts/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    normalized_jsonl_path = out_dir / "datapool.normalized.jsonl"
    normalized_parquet_path = out_dir / "datapool.normalized.parquet"
    train_path = out_dir / "datapool.train.jsonl"
    eval_path = out_dir / "datapool.eval.jsonl"
    sampled_path = out_dir / "datapool.train.sampled.jsonl"
    split_draws_path = out_dir / ".prepare_data.eval_draws.bin"
    intermediate_jsonl_path = out_dir / ".prepare_data.normalized.tmp.jsonl"

    intermediate_format, parquet_row_group_size = _resolve_intermediate_format(data_cfg)
    progress_enabled, progress_every_rows, progress_every_seconds = _resolve_progress_config(
        data_cfg
    )
    normalize_progress = _ProgressReporter(
        phase="normalize",
        enabled=progress_enabled,
        every_rows=progress_every_rows,
        every_seconds=progress_every_seconds,
    )
    split_progress = _ProgressReporter(
        phase="split",
        enabled=progress_enabled,
        every_rows=progress_every_rows,
        every_seconds=progress_every_seconds,
    )
    if intermediate_format == "parquet":
        intermediate_path = normalized_parquet_path
    else:
        intermediate_path = intermediate_jsonl_path

    rng = random.Random(split_seed)
    normalized_rows = 0
    try:
        writer: _ParquetStreamWriter | _JsonlStreamWriter
        if intermediate_format == "parquet":
            writer = _ParquetStreamWriter(
                intermediate_path,
                row_group_size=parquet_row_group_size,
            )
        else:
            writer = _JsonlStreamWriter(intermediate_path)

        with writer as intermediate_writer, split_draws_path.open("wb") as split_draws_writer:
            stream = _iter_raw_rows(data_cfg)
            stream = _iter_normalized_rows(stream, data_cfg)
            stream = _iter_rows_with_length_policy(stream, cfg)
            for row in stream:
                validated = validate_artifact_row(row, "normalized")
                intermediate_writer.write(validated)
                split_draws_writer.write(struct.pack("<d", rng.random()))
                normalized_rows += 1
                normalize_progress.maybe_report(normalized_rows)
        normalize_progress.finish(normalized_rows)

        eval_count = _target_eval_count(normalized_rows, eval_ratio)
        eval_indices = _select_eval_indices_from_draws(
            split_draws_path,
            total_rows=normalized_rows,
            eval_count=eval_count,
        )
        train_rows, eval_rows, sampled_rows = _build_split_artifacts(
            normalized_rows_iter=_iter_intermediate_rows(
                fmt=intermediate_format,
                path=intermediate_path,
                parquet_batch_size=parquet_row_group_size,
            ),
            normalized_output_path=normalized_jsonl_path,
            train_path=train_path,
            eval_path=eval_path,
            sampled_path=sampled_path,
            eval_indices=eval_indices,
            subset_size=subset_limit,
            sampling_strategy=sampling_strategy,
            sampling_seed=sampling_seed,
            on_processed_row=split_progress.maybe_report,
        )
        split_progress.finish(normalized_rows)
    finally:
        if split_draws_path.exists():
            split_draws_path.unlink()
        if intermediate_jsonl_path.exists():
            intermediate_jsonl_path.unlink()

    _write_ood_placeholder(data_cfg, out_dir / "ood_test.jsonl")

    summary = {
        "normalized_rows": normalized_rows,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "sampled_rows": sampled_rows,
        "intermediate_format": intermediate_format,
        "progress_enabled": progress_enabled,
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
