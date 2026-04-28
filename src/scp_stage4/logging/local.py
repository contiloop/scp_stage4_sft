"""Append-only JSONL logger for local SCP artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .schema import (
    RequiredLogContext,
    build_event_record,
    build_failure_record,
    build_metrics_record,
    sanitize_for_log,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - fcntl is available on macOS/Linux.
    fcntl = None  # type: ignore[assignment]


def _lock_file(file_obj: Any) -> None:
    if fcntl is None:
        return
    fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)


def _unlock_file(file_obj: Any) -> None:
    if fcntl is None:
        return
    fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)


def append_jsonl_record(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Append one record to a JSONL file with a file lock."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(sanitize_for_log(dict(record)), ensure_ascii=False, sort_keys=True)
    with file_path.open("a", encoding="utf-8") as handle:
        _lock_file(handle)
        try:
            handle.write(line)
            handle.write("\n")
            handle.flush()
        finally:
            _unlock_file(handle)
    return file_path


class LocalJsonlLogger:
    """Writes run-level and subset-level JSONL logs."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        events_name: str = "events.jsonl",
        metrics_name: str = "metrics.jsonl",
        failures_name: str = "failures.jsonl",
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_name = events_name
        self.metrics_name = metrics_name
        self.failures_name = failures_name

    def _subset_dir(self, subset_idx: int) -> Path:
        return self.run_dir / "subsets" / f"subset_{subset_idx:03d}"

    def _paths_for_kind(self, kind: str, subset_idx: int, *, to_run: bool, to_subset: bool) -> list[Path]:
        if kind not in {"events", "metrics", "failures"}:
            raise ValueError("kind must be one of: events, metrics, failures")
        filename = {
            "events": self.events_name,
            "metrics": self.metrics_name,
            "failures": self.failures_name,
        }[kind]
        paths: list[Path] = []
        if to_run:
            paths.append(self.run_dir / filename)
        if to_subset:
            paths.append(self._subset_dir(subset_idx) / filename)
        return paths

    def _write_record(
        self,
        *,
        kind: str,
        record: Mapping[str, Any],
        subset_idx: int,
        to_run: bool,
        to_subset: bool,
    ) -> list[Path]:
        written: list[Path] = []
        for path in self._paths_for_kind(kind, subset_idx, to_run=to_run, to_subset=to_subset):
            append_jsonl_record(path, record)
            written.append(path)
        return written

    def log_event(
        self,
        *,
        context: RequiredLogContext,
        event_type: str,
        status: str,
        metrics: Mapping[str, Any] | None = None,
        artifact_path: str | None = None,
        error: Any = None,
        extras: Mapping[str, Any] | None = None,
        to_run: bool = True,
        to_subset: bool = True,
    ) -> dict[str, Any]:
        record = build_event_record(
            context=context,
            event_type=event_type,
            status=status,
            metrics=metrics,
            artifact_path=artifact_path,
            error=error,
            extras=extras,
        )
        self._write_record(
            kind="events",
            record=record,
            subset_idx=context.subset_idx,
            to_run=to_run,
            to_subset=to_subset,
        )
        return record

    def log_metrics(
        self,
        *,
        context: RequiredLogContext,
        metrics: Mapping[str, Any],
        status: str = "ok",
        metric_group: str | None = None,
        extras: Mapping[str, Any] | None = None,
        to_run: bool = True,
        to_subset: bool = True,
    ) -> dict[str, Any]:
        record = build_metrics_record(
            context=context,
            metrics=metrics,
            status=status,
            metric_group=metric_group,
            extras=extras,
        )
        self._write_record(
            kind="metrics",
            record=record,
            subset_idx=context.subset_idx,
            to_run=to_run,
            to_subset=to_subset,
        )
        return record

    def log_failure(
        self,
        *,
        context: RequiredLogContext,
        failure_type: str,
        status: str = "failed",
        error: Any = None,
        row_id: str | None = None,
        request_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        attempt: int | None = None,
        extras: Mapping[str, Any] | None = None,
        to_run: bool = True,
        to_subset: bool = True,
    ) -> dict[str, Any]:
        record = build_failure_record(
            context=context,
            failure_type=failure_type,
            status=status,
            error=error,
            row_id=row_id,
            request_id=request_id,
            provider=provider,
            model=model,
            attempt=attempt,
            extras=extras,
        )
        self._write_record(
            kind="failures",
            record=record,
            subset_idx=context.subset_idx,
            to_run=to_run,
            to_subset=to_subset,
        )
        return record
