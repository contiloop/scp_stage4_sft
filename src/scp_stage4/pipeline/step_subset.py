"""Stepwise local subset pipeline with mock/subprocess runtime hooks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import random
import shutil
import subprocess
import sys
import tarfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

_prefetch_log = logging.getLogger(__name__)

from scp_stage4.artifacts import compute_config_hash, persist_effective_config_artifacts
from scp_stage4.config.loader import compose_config
from scp_stage4.config.validator import validate_config
from scp_stage4.data import read_jsonl, validate_row_id_preservation, write_jsonl
from scp_stage4.logging import LocalJsonlLogger, RequiredLogContext
from scp_stage4.schema import QeIsolationRequest, QeIsolationResponse, validate_artifact_rows

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]


class StepSubsetError(RuntimeError):
    """Raised when a stepwise subset contract fails."""


_ARCHIVE_MODE_BY_FORMAT = {
    "tar": "w",
    "tar.gz": "w:gz",
    "tar.xz": "w:xz",
}
_ARCHIVE_SUFFIX_BY_FORMAT = {
    "tar": ".tar",
    "tar.gz": ".tar.gz",
    "tar.xz": ".tar.xz",
}


def _get_by_dotpath(cfg: Mapping[str, Any], key: str, default: Any = None) -> Any:
    cursor: Any = cfg
    for part in key.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _subset_dir(run_root: Path, subset_idx: int) -> Path:
    return run_root / "subsets" / f"subset_{subset_idx:03d}"


def _as_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _read_artifact(path: Path, artifact_name: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise StepSubsetError(f"Missing required artifact: {path}")
    rows = _as_rows(read_jsonl(path))
    return validate_artifact_rows(rows, artifact_name)


def _write_artifact(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    artifact_name: str,
) -> list[dict[str, Any]]:
    normalized = validate_artifact_rows(rows, artifact_name)
    write_jsonl(path, normalized, ensure_ascii=False)
    return normalized


def _iter_parquet_mapping_rows(
    path: Path,
    *,
    batch_size: int = 4096,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    if pa is None or pq is None:
        return []
    row_cap = None if max_rows is None else max(1, int(max_rows))
    parquet_file = pq.ParquetFile(str(path))
    rows: list[dict[str, Any]] = []
    for record_batch in parquet_file.iter_batches(batch_size=max(1, int(batch_size))):
        table = pa.Table.from_batches([record_batch])
        for row in table.to_pylist():
            if isinstance(row, Mapping):
                rows.append(dict(row))
                if row_cap is not None and len(rows) >= row_cap:
                    return rows
    return rows


def _iter_jsonl_rows(path: Path, *, max_rows: int | None = None) -> list[dict[str, Any]]:
    row_cap = None if max_rows is None else max(1, int(max_rows))
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if isinstance(parsed, Mapping):
                rows.append(dict(parsed))
                if row_cap is not None and len(rows) >= row_cap:
                    return rows
    return rows


def _load_prepared_rows(path: Path, *, max_rows: int | None = None) -> list[dict[str, Any]]:
    try:
        if path.suffix.lower() == ".parquet":
            rows = _iter_parquet_mapping_rows(path, max_rows=max_rows)
        else:
            if max_rows is None:
                rows = _as_rows(read_jsonl(path))
            else:
                rows = _iter_jsonl_rows(path, max_rows=max_rows)
    except Exception as exc:
        raise StepSubsetError(f"failed to load prepared rows from {path}: {exc}") from exc
    if not rows:
        return []
    return validate_artifact_rows(rows, "normalized")


def _subset_size_hint(cfg: Mapping[str, Any], subset_size_override: int | None) -> int | None:
    subset_size = subset_size_override
    if subset_size is None:
        subset_size = _get_by_dotpath(cfg, "data.subset_size")
    if subset_size is None:
        strategy = str(_get_by_dotpath(cfg, "pipeline.subset.strategy", "fraction"))
        if strategy == "fixed_size":
            subset_size = _get_by_dotpath(cfg, "pipeline.subset.fixed_size")
        else:
            return None
    max_size = _get_by_dotpath(cfg, "pipeline.subset.max_size")
    if max_size is not None:
        subset_size = min(int(subset_size), int(max_size))
    return max(1, int(subset_size))


def _load_fixture_rows() -> list[dict[str, Any]]:
    candidates = [
        Path("tests/fixtures/datapool.train.jsonl"),
        Path("tests/fixtures/input.jsonl"),
        Path("tests/fixtures/input.happy.jsonl"),
    ]
    for path in candidates:
        if path.exists():
            rows = _as_rows(read_jsonl(path))
            if rows:
                return validate_artifact_rows(rows, "normalized")

    rows: list[dict[str, Any]] = []
    for idx in range(64):
        rows.append(
            {
                "id": f"row_{idx:04d}",
                "dataset": "local_fixture",
                "source": f"Source sentence {idx}",
                "metadata": {
                    "title": None,
                    "document_type": "other",
                    "text_role": "body",
                    "original_id": str(idx),
                    "parent_id": None,
                    "chunk_idx": None,
                },
            }
        )
    return validate_artifact_rows(rows, "normalized")


def _select_subset(
    rows: list[dict[str, Any]],
    cfg: Mapping[str, Any],
    subset_idx: int,
    subset_size_override: int | None,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    seed = int(_get_by_dotpath(cfg, "pipeline.subset.seed", 42))
    shuffled = list(rows)
    if bool(_get_by_dotpath(cfg, "pipeline.subset.shuffle", True)):
        rng = random.Random(seed)
        rng.shuffle(shuffled)

    subset_size = subset_size_override
    if subset_size is None:
        subset_size = _get_by_dotpath(cfg, "data.subset_size")
    if subset_size is None:
        strategy = str(_get_by_dotpath(cfg, "pipeline.subset.strategy", "fraction"))
        if strategy == "fixed_size":
            subset_size = _get_by_dotpath(cfg, "pipeline.subset.fixed_size")
        else:
            fraction = float(_get_by_dotpath(cfg, "pipeline.subset.fraction", 0.02))
            min_size = int(_get_by_dotpath(cfg, "pipeline.subset.min_size", 32))
            subset_size = max(min_size, int(len(shuffled) * fraction + 0.999999))

    max_size = _get_by_dotpath(cfg, "pipeline.subset.max_size")
    if max_size is not None:
        subset_size = min(int(subset_size), int(max_size))

    size = max(1, int(subset_size))
    start = subset_idx * size
    end = start + size
    if start >= len(shuffled):
        return []

    window = shuffled[start:end]
    drop_last = bool(_get_by_dotpath(cfg, "pipeline.subset.drop_last", False))
    if drop_last and len(window) < size:
        return []
    return window


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _zscore(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std <= 0:
        return [0.0 for _ in values]
    return [(value - mean) / std for value in values]


def _qe_score_transform(ctx: PipelineContext) -> tuple[str, str, float, bool]:
    score_direction = str(_get_by_dotpath(ctx.cfg, "qe.primary.score_direction", "higher_is_better"))
    if score_direction not in {"higher_is_better", "lower_is_better"}:
        raise StepSubsetError(
            "qe.primary.score_direction must be 'higher_is_better' or 'lower_is_better'"
        )
    transform_cfg = _get_by_dotpath(ctx.cfg, "qe.primary.transform", {})
    if not isinstance(transform_cfg, Mapping):
        transform_cfg = {}

    transform_type = str(transform_cfg.get("type", "invert" if score_direction == "lower_is_better" else "none"))
    max_score = float(transform_cfg.get("max_score", 25.0))
    clamp_for_quality = bool(
        transform_cfg.get("clamp_for_quality", transform_type == "invert")
    )
    if transform_type not in {"none", "invert"}:
        raise StepSubsetError("qe.primary.transform.type must be 'none' or 'invert'")
    if max_score <= 0:
        raise StepSubsetError("qe.primary.transform.max_score must be > 0")
    return score_direction, transform_type, max_score, clamp_for_quality


def _qe_quality_from_raw(
    *,
    ctx: PipelineContext,
    raw_score: float,
) -> tuple[float, bool]:
    score_direction, transform_type, max_score, clamp_for_quality = _qe_score_transform(ctx)
    if transform_type == "invert":
        if clamp_for_quality:
            clamped_raw = _clamp(raw_score, 0.0, max_score)
            metricx_clamped = not math.isclose(clamped_raw, raw_score, rel_tol=0.0, abs_tol=1e-12)
        else:
            clamped_raw = raw_score
            metricx_clamped = False
        return max_score - clamped_raw, metricx_clamped
    if score_direction == "lower_is_better":
        return -raw_score, False
    return raw_score, False


@dataclass(frozen=True)
class PipelineContext:
    cfg: dict[str, Any]
    cfg_hash: str
    run_id: str
    subset_idx: int
    run_root: Path
    subset_root: Path
    logger: LocalJsonlLogger


@dataclass(frozen=True)
class SubsetArchiveConfig:
    enabled: bool
    format: str
    output_dir: str
    delete_original_after_archive: bool


def _build_context(
    *,
    config_path: str,
    overrides: list[str] | None,
    run_id_override: str | None,
    subset_idx: int,
) -> PipelineContext:
    cfg = compose_config(config_path, overrides=overrides)
    validate_config(cfg)

    run_id = run_id_override or str(_get_by_dotpath(cfg, "run.run_id", "local_contract"))
    root_dir = Path(str(_get_by_dotpath(cfg, "logging.local.root_dir", "artifacts/runs")))
    run_root = root_dir / run_id
    subset_root = _subset_dir(run_root, subset_idx)
    subset_root.mkdir(parents=True, exist_ok=True)

    cfg_hash = compute_config_hash(cfg)
    persisted = persist_effective_config_artifacts(
        run_dir=run_root,
        effective_config=cfg,
        write_effective_config=bool(
            _get_by_dotpath(cfg, "logging.local.write_effective_config", True)
        ),
        write_config_hash=bool(_get_by_dotpath(cfg, "logging.local.write_config_hash", True)),
    )
    if str(persisted["config_hash"]) != cfg_hash:
        raise StepSubsetError("config_hash mismatch between stable hash and persisted hash")

    local_cfg = _get_by_dotpath(cfg, "logging.local", {})
    logger = LocalJsonlLogger(
        run_root,
        events_name=str(local_cfg.get("events_jsonl", "events.jsonl")),
        metrics_name=str(local_cfg.get("metrics_jsonl", "metrics.jsonl")),
        failures_name=str(local_cfg.get("failures_jsonl", "failures.jsonl")),
    )

    _maybe_init_weave(cfg, run_id=run_id)

    return PipelineContext(
        cfg=cfg,
        cfg_hash=cfg_hash,
        run_id=run_id,
        subset_idx=subset_idx,
        run_root=run_root,
        subset_root=subset_root,
        logger=logger,
    )


_weave_initialized: bool = False


def _maybe_init_weave(cfg: Mapping[str, Any], *, run_id: str) -> None:
    global _weave_initialized
    if _weave_initialized:
        return
    weave_cfg = _get_by_dotpath(cfg, "logging.weave", {}) or {}
    if not weave_cfg.get("enabled", False):
        return
    try:
        import weave  # type: ignore[import-untyped]

        project = str(weave_cfg.get("project", "scp_main"))
        weave.init(project)
        if weave_cfg.get("implicitly_patch_integrations", True):
            try:
                import weave.integrations.transformers  # type: ignore[import-untyped]  # noqa: F401
            except Exception:
                pass
        _weave_initialized = True
    except Exception as exc:
        _prefetch_log.warning("weave init skipped: %s", exc)


def _context_for_phase(ctx: PipelineContext, phase: str) -> RequiredLogContext:
    return RequiredLogContext(
        run_id=ctx.run_id,
        subset_idx=ctx.subset_idx,
        phase=phase,
        config_hash=ctx.cfg_hash,
    )


def _touch_failure_layout(ctx: PipelineContext) -> None:
    failures_name = str(_get_by_dotpath(ctx.cfg, "logging.local.failures_jsonl", "failures.jsonl"))
    (ctx.run_root / failures_name).touch(exist_ok=True)
    (ctx.subset_root / failures_name).touch(exist_ok=True)


def _subset_archive_config(cfg: Mapping[str, Any]) -> SubsetArchiveConfig:
    raw = _get_by_dotpath(cfg, "pipeline.stage.subset_archive", {})
    if not isinstance(raw, Mapping):
        raw = {}
    format_name = str(raw.get("format", "tar.gz"))
    if format_name not in _ARCHIVE_MODE_BY_FORMAT:
        raise StepSubsetError(
            "pipeline.stage.subset_archive.format must be one of: tar, tar.gz, tar.xz"
        )
    output_dir = str(raw.get("output_dir", "archives/subsets")).strip()
    if not output_dir:
        raise StepSubsetError("pipeline.stage.subset_archive.output_dir must be non-empty")
    return SubsetArchiveConfig(
        enabled=bool(raw.get("enabled", False)),
        format=format_name,
        output_dir=output_dir,
        delete_original_after_archive=bool(raw.get("delete_original_after_archive", False)),
    )


def _subset_archive_paths(
    *,
    run_root: Path,
    subset_idx: int,
    archive_cfg: SubsetArchiveConfig,
) -> tuple[Path, Path]:
    stem = f"subset_{subset_idx:03d}"
    archive_root = run_root / archive_cfg.output_dir
    suffix = _ARCHIVE_SUFFIX_BY_FORMAT[archive_cfg.format]
    return archive_root / f"{stem}{suffix}", archive_root / f"{stem}.manifest.json"


def _subset_inventory(subset_root: Path) -> list[str]:
    files = [path for path in subset_root.rglob("*") if path.is_file()]
    return sorted(str(path.relative_to(subset_root)) for path in files)


def _archive_subset_if_configured(
    *,
    ctx: PipelineContext,
    stage_completed: bool,
    counts: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    archive_cfg = _subset_archive_config(ctx.cfg)
    if not archive_cfg.enabled:
        return None

    subset_root = ctx.subset_root
    if not subset_root.exists():
        raise StepSubsetError(f"subset root missing for archive: {subset_root}")

    inventory = _subset_inventory(subset_root)
    if not inventory:
        raise StepSubsetError(f"subset root has no files to archive: {subset_root}")

    archive_path, manifest_path = _subset_archive_paths(
        run_root=ctx.run_root,
        subset_idx=ctx.subset_idx,
        archive_cfg=archive_cfg,
    )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()

    with tarfile.open(archive_path, _ARCHIVE_MODE_BY_FORMAT[archive_cfg.format]) as handle:
        handle.add(subset_root, arcname=f"subset_{ctx.subset_idx:03d}")

    manifest = {
        "status": "ok",
        "run_id": ctx.run_id,
        "subset_idx": ctx.subset_idx,
        "config_hash": ctx.cfg_hash,
        "format": archive_cfg.format,
        "archive_path": str(archive_path),
        "subset_path": str(subset_root),
        "file_count": len(inventory),
        "files": inventory,
        "counts": dict(counts) if counts is not None else None,
    }
    _write_json_file(manifest_path, manifest)

    deleted_original = False
    if archive_cfg.delete_original_after_archive and stage_completed:
        shutil.rmtree(subset_root)
        subset_root.mkdir(parents=True, exist_ok=True)
        _write_json_file(
            subset_root / "ARCHIVED.json",
            {
                "status": "archived",
                "run_id": ctx.run_id,
                "subset_idx": ctx.subset_idx,
                "config_hash": ctx.cfg_hash,
                "archive_path": str(archive_path),
                "manifest_path": str(manifest_path),
            },
        )
        deleted_original = True

    archive_rel = str(archive_path.relative_to(ctx.run_root))
    manifest_rel = str(manifest_path.relative_to(ctx.run_root))
    archive_bytes = archive_path.stat().st_size
    ctx.logger.log_event(
        context=_context_for_phase(ctx, "archive-subset"),
        event_type="phase_completed",
        status="ok",
        artifact_path=archive_rel,
        metrics={
            "archived_file_count": len(inventory),
            "archive_bytes": archive_bytes,
            "deleted_original_subset_dir": 1 if deleted_original else 0,
        },
        extras={"manifest_path": manifest_rel},
    )
    ctx.logger.log_metrics(
        context=_context_for_phase(ctx, "archive-subset"),
        metrics={
            "subset/archive_file_count": len(inventory),
            "subset/archive_size_bytes": archive_bytes,
            "subset/archive_deleted_original": 1 if deleted_original else 0,
        },
        metric_group="subset",
    )
    _touch_failure_layout(ctx)

    return {
        "enabled": True,
        "archive_path": str(archive_path),
        "manifest_path": str(manifest_path),
        "file_count": len(inventory),
        "archive_size_bytes": archive_bytes,
        "deleted_original_subset_dir": deleted_original,
    }


def _finalize_stage_archive_cleanup(
    *,
    ctx: PipelineContext,
    subset_indices: Sequence[int],
) -> int:
    archive_cfg = _subset_archive_config(ctx.cfg)
    if not archive_cfg.enabled or not archive_cfg.delete_original_after_archive:
        return 0

    deleted_count = 0
    for subset_idx in subset_indices:
        archive_path, manifest_path = _subset_archive_paths(
            run_root=ctx.run_root,
            subset_idx=subset_idx,
            archive_cfg=archive_cfg,
        )
        if not archive_path.exists() or not manifest_path.exists():
            raise StepSubsetError(
                "subset archive cleanup requires existing archive+manifest; "
                f"missing for subset_{subset_idx:03d}"
            )
        subset_root = _subset_dir(ctx.run_root, subset_idx)
        if subset_root.exists():
            shutil.rmtree(subset_root)
        subset_root.mkdir(parents=True, exist_ok=True)
        _write_json_file(
            subset_root / "ARCHIVED.json",
            {
                "status": "archived",
                "run_id": ctx.run_id,
                "subset_idx": subset_idx,
                "config_hash": ctx.cfg_hash,
                "archive_path": str(archive_path),
                "manifest_path": str(manifest_path),
            },
        )
        context = RequiredLogContext(
            run_id=ctx.run_id,
            subset_idx=subset_idx,
            phase="archive-subset",
            config_hash=ctx.cfg_hash,
        )
        ctx.logger.log_event(
            context=context,
            event_type="subset_archive_pruned",
            status="ok",
            artifact_path=str(archive_path.relative_to(ctx.run_root)),
            extras={"manifest_path": str(manifest_path.relative_to(ctx.run_root))},
        )
        deleted_count += 1

    return deleted_count


def _log_cli_failure(
    *,
    config_path: str,
    overrides: list[str] | None,
    run_id_override: str | None,
    subset_idx: int,
    phase: str,
    failure: Exception,
) -> None:
    try:
        ctx = _build_context(
            config_path=config_path,
            overrides=overrides,
            run_id_override=run_id_override,
            subset_idx=subset_idx,
        )
        _touch_failure_layout(ctx)
        context = _context_for_phase(ctx, phase)
        ctx.logger.log_failure(
            context=context,
            failure_type=f"{phase}_failed",
            status="failed",
            error=str(failure),
        )
        ctx.logger.log_event(
            context=context,
            event_type="phase_failed",
            status="failed",
            error=str(failure),
        )
    except Exception:
        # Best-effort failure logging: preserve original exit behavior if logging setup fails.
        return


def _runtime_mode(ctx: PipelineContext, section: str) -> str:
    return str(_get_by_dotpath(ctx.cfg, f"{section}.runtime.mode", "mock"))


def _subprocess_command(ctx: PipelineContext, section: str) -> list[str]:
    raw = _get_by_dotpath(ctx.cfg, f"{section}.runtime.subprocess.command", None)
    if not isinstance(raw, list) or not raw:
        raise StepSubsetError(
            f"{section}.runtime.subprocess.command must be a non-empty list when mode=subprocess"
        )
    command: list[str] = []
    for part in raw:
        if not isinstance(part, str) or not part.strip():
            raise StepSubsetError(
                f"{section}.runtime.subprocess.command contains non-string/empty part: {part!r}"
            )
        command.append(part)
    return command


def _subprocess_context_args(
    *,
    ctx: PipelineContext,
    section: str,
    phase: str,
) -> list[str]:
    return [
        "--effective-config",
        str(ctx.run_root / "effective_config.yaml"),
        "--config-hash",
        ctx.cfg_hash,
        "--run-id",
        ctx.run_id,
        "--subset-idx",
        str(ctx.subset_idx),
        "--section",
        section,
        "--phase",
        phase,
    ]


def _run_subprocess_jsonl(
    *,
    ctx: PipelineContext,
    section: str,
    phase: str,
    input_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    command = _subprocess_command(ctx, section)

    runtime_dir = ctx.subset_root / "runtime_io"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    input_path = runtime_dir / f"{phase}.input.jsonl"
    output_path = runtime_dir / f"{phase}.output.jsonl"
    write_jsonl(input_path, input_rows, ensure_ascii=False)

    cmd = (
        list(command)
        + ["--input", str(input_path), "--output", str(output_path)]
        + _subprocess_context_args(ctx=ctx, section=section, phase=phase)
    )
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None, text=True)
    if result.returncode != 0:
        stdout = (result.stdout or "").strip()
        detail = stdout or "no output (check stderr above)"
        raise StepSubsetError(f"{section} subprocess failed ({result.returncode}): {detail}")
    if not output_path.exists():
        raise StepSubsetError(f"{section} subprocess did not produce output JSONL: {output_path}")

    return _as_rows(read_jsonl(output_path))


def _training_runtime_mode(ctx: PipelineContext) -> str:
    return str(_get_by_dotpath(ctx.cfg, "training.runtime.mode", "mock"))


def _training_subprocess_command(ctx: PipelineContext, command_key: str) -> list[str]:
    raw = _get_by_dotpath(ctx.cfg, f"training.runtime.subprocess.{command_key}", None)
    if not isinstance(raw, list) or not raw:
        raise StepSubsetError(
            f"training.runtime.subprocess.{command_key} must be a non-empty list "
            "when training.runtime.mode=subprocess"
        )
    command: list[str] = []
    for part in raw:
        if not isinstance(part, str) or not part.strip():
            raise StepSubsetError(
                f"training.runtime.subprocess.{command_key} contains invalid part: {part!r}"
            )
        command.append(part)
    return command


def _run_training_subprocess_jsonl(
    *,
    ctx: PipelineContext,
    command_key: str,
    phase: str,
    input_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    command = _training_subprocess_command(ctx, command_key)

    runtime_dir = ctx.subset_root / "runtime_io"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    input_path = runtime_dir / f"{phase}.input.jsonl"
    output_path = runtime_dir / f"{phase}.output.jsonl"
    write_jsonl(input_path, input_rows, ensure_ascii=False)

    cmd = (
        list(command)
        + ["--input", str(input_path), "--output", str(output_path)]
        + _subprocess_context_args(ctx=ctx, section="training", phase=phase)
    )
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None, text=True)
    if result.returncode != 0:
        stdout = (result.stdout or "").strip()
        detail = stdout or "no output (check stderr above)"
        raise StepSubsetError(f"training subprocess failed ({result.returncode}): {detail}")
    if not output_path.exists():
        raise StepSubsetError(f"training subprocess did not produce output JSONL: {output_path}")
    return _as_rows(read_jsonl(output_path))


def _validate_status_rows(rows: Sequence[Mapping[str, Any]], *, phase: str) -> None:
    if not rows:
        raise StepSubsetError(f"{phase} subprocess produced no status rows")
    for idx, row in enumerate(rows):
        status = row.get("status", "ok")
        if status != "ok":
            raise StepSubsetError(
                f"{phase} subprocess status row {idx} failed: {row.get('error')}"
            )


def _normalize_clean_base_evidence(
    *,
    status_row: Mapping[str, Any],
    collapse_adapter: str,
    strict: bool,
) -> dict[str, Any]:
    clean_base = status_row.get("clean_base")
    if clean_base is None and not strict:
        clean_base = True
    if clean_base is not True:
        raise StepSubsetError("unload-collapse-lora evidence missing clean_base=true")

    active_adapters = status_row.get("active_adapters")
    if active_adapters is None and not strict:
        active_adapters = []
    if not isinstance(active_adapters, list):
        raise StepSubsetError("unload-collapse-lora evidence.active_adapters must be a list")
    if active_adapters:
        raise StepSubsetError(
            f"unload-collapse-lora evidence has active adapters after unload: {active_adapters}"
        )

    collapse_merged = status_row.get("collapse_merged")
    if collapse_merged is None and not strict:
        collapse_merged = False
    if collapse_merged is not False:
        raise StepSubsetError("unload-collapse-lora evidence must report collapse_merged=false")

    adapter_registry_hash = status_row.get("adapter_registry_hash")
    if not isinstance(adapter_registry_hash, str) or not adapter_registry_hash.strip():
        if strict:
            raise StepSubsetError(
                "unload-collapse-lora evidence missing adapter_registry_hash in subprocess mode"
            )
        adapter_registry_hash = hashlib.sha256(collapse_adapter.encode("utf-8")).hexdigest()

    verified_adapter_path = status_row.get("verified_adapter_path")
    if verified_adapter_path is None and not strict:
        verified_adapter_path = collapse_adapter
    if not isinstance(verified_adapter_path, str) or not verified_adapter_path.strip():
        raise StepSubsetError(
            "unload-collapse-lora evidence.verified_adapter_path must be a non-empty string"
        )

    return {
        "clean_base": True,
        "active_adapters": [],
        "collapse_merged": False,
        "adapter_registry_hash": str(adapter_registry_hash),
        "verified_adapter_path": str(verified_adapter_path),
    }


def _collapse_state_path(ctx: PipelineContext) -> Path:
    return ctx.subset_root / "collapse_adapter" / "collapse_state.json"


def _clean_base_state_path(ctx: PipelineContext) -> Path:
    return ctx.subset_root / "clean_base.json"


def _latest_checkpoint_path(ctx: PipelineContext) -> Path:
    return ctx.run_root / "checkpoints" / "latest.json"


def _read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise StepSubsetError(f"Expected JSON object at {path}")
    return loaded


def _write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _latest_checkpoint_ref(ctx: PipelineContext) -> str | None:
    state = _read_json_file(_latest_checkpoint_path(ctx))
    if not state:
        return None
    value = state.get("checkpoint_path")
    return str(value) if value is not None else None


def _checkpoint_retention_keep_last_n(cfg: Mapping[str, Any]) -> int:
    raw = _get_by_dotpath(cfg, "training.checkpoint.keep_last_n", 2)
    if isinstance(raw, bool) or not isinstance(raw, int):
        return 2
    return max(1, raw)


def _checkpoint_retention_keep_best_n(cfg: Mapping[str, Any]) -> int:
    raw = _get_by_dotpath(cfg, "training.checkpoint.keep_best_n", 1)
    if isinstance(raw, bool) or not isinstance(raw, int):
        return 1
    return max(0, raw)


def _checkpoint_retention_metric_for_best(cfg: Mapping[str, Any]) -> str:
    raw = _get_by_dotpath(cfg, "training.checkpoint.metric_for_best", "ood/metricx24_ref_quality_mean")
    if not isinstance(raw, str) or not raw.strip():
        return "ood/metricx24_ref_quality_mean"
    return raw.strip()


def _checkpoint_retention_greater_is_better(cfg: Mapping[str, Any]) -> bool:
    raw = _get_by_dotpath(cfg, "training.checkpoint.greater_is_better", True)
    return bool(raw)


def _metrics_jsonl_path(ctx: PipelineContext) -> Path:
    filename = str(_get_by_dotpath(ctx.cfg, "logging.local.metrics_jsonl", "metrics.jsonl"))
    return ctx.run_root / filename


def _best_subset_indices_for_retention(
    *,
    ctx: PipelineContext,
    upto_subset_idx: int,
) -> list[int]:
    keep_best_n = _checkpoint_retention_keep_best_n(ctx.cfg)
    if keep_best_n <= 0:
        return []
    metrics_path = _metrics_jsonl_path(ctx)
    if not metrics_path.exists():
        return []

    metric_key = _checkpoint_retention_metric_for_best(ctx.cfg)
    greater_is_better = _checkpoint_retention_greater_is_better(ctx.cfg)
    score_by_subset: dict[int, float] = {}

    with metrics_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, Mapping):
                continue
            phase = parsed.get("phase")
            if phase != "eval-ood":
                continue
            subset_value = parsed.get("subset_idx")
            if isinstance(subset_value, bool) or not isinstance(subset_value, int):
                continue
            subset_idx = int(subset_value)
            if subset_idx >= upto_subset_idx:
                continue
            metrics = parsed.get("metrics")
            if not isinstance(metrics, Mapping):
                continue
            metric_value = metrics.get(metric_key)
            if isinstance(metric_value, bool) or not isinstance(metric_value, (int, float)):
                continue
            score = float(metric_value)
            if not math.isfinite(score):
                continue
            score_by_subset[subset_idx] = score

    if not score_by_subset:
        return []

    sorted_items = sorted(
        score_by_subset.items(),
        key=lambda item: (item[1], item[0]),
        reverse=greater_is_better,
    )
    return [subset_idx for subset_idx, _ in sorted_items[:keep_best_n]]


def _path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += int(child.stat().st_size)
    return total


def _prune_subset_checkpoints_if_configured(ctx: PipelineContext) -> dict[str, int]:
    current_subset_idx = int(ctx.subset_idx)
    keep_last_n = _checkpoint_retention_keep_last_n(ctx.cfg)
    keep_best_n = _checkpoint_retention_keep_best_n(ctx.cfg)
    metric_for_best = _checkpoint_retention_metric_for_best(ctx.cfg)
    greater_is_better = _checkpoint_retention_greater_is_better(ctx.cfg)

    preserve_previous = set(range(max(0, current_subset_idx - keep_last_n), current_subset_idx))
    preserve_best = set(
        _best_subset_indices_for_retention(ctx=ctx, upto_subset_idx=current_subset_idx)
    )
    preserve_subset_indices = set(preserve_previous)
    preserve_subset_indices.update(preserve_best)
    preserve_subset_indices.add(current_subset_idx)

    subset_root = ctx.run_root / "subsets"
    if not subset_root.exists():
        return {
            "subset_count": 0,
            "deleted_count": 0,
            "freed_bytes": 0,
            "preserved_subset_count": len(preserve_subset_indices),
            "preserved_best_count": len(preserve_best),
        }

    preserve_names = {
        "train_rows.jsonl",
        "checkpoint_state.json",
        "worker_checkpoint_state.json",
        "PRUNED_CHECKPOINTS.json",
    }
    subset_count = 0
    deleted_count = 0
    freed_bytes = 0

    for subset_dir in sorted(subset_root.glob("subset_*")):
        if not subset_dir.is_dir():
            continue
        try:
            subset_num = int(subset_dir.name.split("_")[-1])
        except ValueError:
            continue
        if subset_num in preserve_subset_indices or subset_num >= current_subset_idx:
            continue
        train_final_dir = subset_dir / "train_final"
        if not train_final_dir.exists() or not train_final_dir.is_dir():
            continue

        removed: list[str] = []
        for child in sorted(train_final_dir.iterdir()):
            if child.name in preserve_names:
                continue
            freed_bytes += _path_size_bytes(child)
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
            removed.append(child.name)
            deleted_count += 1
        if not removed:
            continue

        subset_count += 1
        _write_json_file(
            train_final_dir / "PRUNED_CHECKPOINTS.json",
            {
                "status": "ok",
                "run_id": ctx.run_id,
                "subset_idx": subset_num,
                "retention_keep_last_n": keep_last_n,
                "retention_keep_best_n": keep_best_n,
                "retention_metric_for_best": metric_for_best,
                "retention_greater_is_better": greater_is_better,
                "preserved_subset_indices": sorted(preserve_subset_indices),
                "deleted_entries": removed,
            },
        )

    return {
        "subset_count": subset_count,
        "deleted_count": deleted_count,
        "freed_bytes": freed_bytes,
        "preserved_subset_count": len(preserve_subset_indices),
        "preserved_best_count": len(preserve_best),
    }


def _collapse_adapter_ref(ctx: PipelineContext) -> str:
    state = _read_json_file(_collapse_state_path(ctx))
    if not state or state.get("status") != "ok":
        raise StepSubsetError(
            "collapse adapter state is missing; run train-collapse-lora before infer-q2"
        )
    adapter_path = state.get("adapter_path")
    if not isinstance(adapter_path, str) or not adapter_path.strip():
        raise StepSubsetError("collapse adapter state missing adapter_path")
    return adapter_path


def _assert_clean_base(ctx: PipelineContext) -> None:
    state = _read_json_file(_clean_base_state_path(ctx))
    if not state or state.get("status") != "ok":
        raise StepSubsetError(
            "clean base verification is missing; run unload-collapse-lora before API/update"
        )
    if state.get("clean_base") is not True:
        raise StepSubsetError("clean base verification missing clean_base=true")
    active_adapters = state.get("active_adapters")
    if not isinstance(active_adapters, list) or active_adapters:
        raise StepSubsetError("clean base verification must report no active adapters")
    if state.get("collapse_merged") is not False:
        raise StepSubsetError("clean base verification must report collapse_merged=false")
    registry_hash = state.get("adapter_registry_hash")
    if not isinstance(registry_hash, str) or not registry_hash.strip():
        raise StepSubsetError("clean base verification missing adapter_registry_hash")


def _materialize_input_rows(
    ctx: PipelineContext,
    *,
    subset_size_override: int | None,
    use_prepared_data: bool,
    use_sampled_data: bool,
) -> list[dict[str, Any]]:
    input_path = ctx.subset_root / "input.jsonl"

    # Reuse prefetched file if already written (e.g. by background prefetch).
    if input_path.exists() and input_path.stat().st_size > 0:
        try:
            existing = validate_artifact_rows(_as_rows(read_jsonl(input_path)), "normalized")
            if existing:
                return existing
        except Exception:
            pass  # corrupted prefetch — fall through and re-materialize

    pool_rows: list[dict[str, Any]] = []
    if use_prepared_data:
        load_limit: int | None = None
        shuffle = bool(_get_by_dotpath(ctx.cfg, "pipeline.subset.shuffle", True))
        if not shuffle:
            size_hint = _subset_size_hint(ctx.cfg, subset_size_override=subset_size_override)
            if size_hint is not None:
                load_limit = (ctx.subset_idx + 1) * size_hint
        prepared_candidates = []
        if use_sampled_data:
            prepared_candidates.append(Path("artifacts/data/datapool.train.sampled.parquet"))
            prepared_candidates.append(Path("artifacts/data/datapool.train.sampled.jsonl"))
        prepared_candidates.append(Path("artifacts/data/datapool.train.parquet"))
        prepared_candidates.append(Path("artifacts/data/datapool.train.jsonl"))
        for candidate in prepared_candidates:
            if candidate.exists():
                loaded = _load_prepared_rows(candidate, max_rows=load_limit)
                if loaded:
                    pool_rows = loaded
                    break

    if use_prepared_data and not pool_rows:
        raise StepSubsetError(
            "No prepared train rows found; run prepare-data before using prepared-data mode"
        )

    if not pool_rows:
        pool_rows = _load_fixture_rows()

    selected_rows = _select_subset(pool_rows, ctx.cfg, ctx.subset_idx, subset_size_override)
    if not selected_rows:
        raise StepSubsetError("No rows available to build subset input")

    return _write_artifact(input_path, selected_rows, "input")


def _generate_mt_rows(
    *,
    ctx: PipelineContext,
    rows: Sequence[Mapping[str, Any]],
    q_tag: str,
) -> list[dict[str, Any]]:
    mt_key = f"mt_{q_tag}"
    mode = _runtime_mode(ctx, "inference")

    if mode == "mock":
        out_rows: list[dict[str, Any]] = []
        for row in rows:
            out = dict(row)
            if q_tag == "q1":
                out[mt_key] = f"KO_Q1::{row['id']}"
            else:
                out[mt_key] = f"KO_Q2::{row['id']}"
            out_rows.append(out)
        return out_rows

    if mode == "subprocess":
        base_checkpoint = _latest_checkpoint_ref(ctx)
        collapse_adapter = _collapse_adapter_ref(ctx) if q_tag == "q2" else None
        requests = [
            {
                "id": f"{ctx.run_id}/subsets/subset_{ctx.subset_idx:03d}/{row['id']}/{q_tag}",
                "run_id": ctx.run_id,
                "subset_idx": ctx.subset_idx,
                "row_id": row["id"],
                "q_tag": q_tag,
                "source": row["source"],
                "metadata": row.get("metadata", {}),
                "base_checkpoint": base_checkpoint,
                "collapse_adapter": collapse_adapter,
                "decoding": _get_by_dotpath(ctx.cfg, f"inference.{q_tag}", {}),
                "runtime_config": {
                    "model": _get_by_dotpath(ctx.cfg, "model", {}),
                    "inference": _get_by_dotpath(ctx.cfg, "inference", {}),
                    "data_length": _get_by_dotpath(ctx.cfg, "data.length", {}),
                },
            }
            for row in rows
        ]
        response_rows = _run_subprocess_jsonl(
            ctx=ctx,
            section="inference",
            phase=f"infer-{q_tag}",
            input_rows=requests,
        )

        by_id: dict[str, dict[str, Any]] = {}
        for resp in response_rows:
            resp_id = resp.get("id")
            mt = resp.get("mt")
            status = resp.get("status", "ok")
            if not isinstance(resp_id, str) or not resp_id:
                raise StepSubsetError("inference subprocess response missing id")
            if status != "ok":
                error = resp.get("error")
                raise StepSubsetError(
                    f"inference subprocess row failed for id={resp_id}: {error}"
                )
            if not isinstance(mt, str) or not mt.strip():
                raise StepSubsetError(f"inference subprocess response missing mt for id={resp_id}")
            by_id[resp_id] = resp

        out_rows = []
        for req, row in zip(requests, rows):
            resp = by_id.get(str(req["id"]))
            if resp is None:
                raise StepSubsetError(
                    f"inference subprocess missing response for request id={req['id']}"
                )
            out = dict(row)
            out[mt_key] = str(resp["mt"])
            out_rows.append(out)
        return out_rows

    raise StepSubsetError(f"Unsupported inference runtime mode: {mode}")


def _score_mt_rows(
    *,
    ctx: PipelineContext,
    rows: Sequence[Mapping[str, Any]],
    q_tag: str,
) -> list[dict[str, Any]]:
    mode = _runtime_mode(ctx, "qe")
    score_direction, _, _, _ = _qe_score_transform(ctx)
    backend = str(_get_by_dotpath(ctx.cfg, "qe.primary.backend", "metricx24"))

    if mode == "mock":
        score_rows: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            if q_tag == "q1":
                raw_score = round(0.90 - (idx % 5) * 0.07, 6)
            else:
                collapse_drop = round(0.03 + (idx % 4) * 0.04, 6)
                if score_direction == "lower_is_better":
                    qe_q1_raw = float(row.get("qe_raw_q1", row.get("qe_q1", 0.0)))
                    raw_score = round(qe_q1_raw + collapse_drop, 6)
                else:
                    qe_q1 = float(row.get("qe_q1", 0.0))
                    raw_score = round(max(0.0, qe_q1 - collapse_drop), 6)
            if not math.isfinite(raw_score):
                raise StepSubsetError(f"mock qe produced non-finite raw score for q_tag={q_tag}")
            quality_score, metricx_clamped = _qe_quality_from_raw(ctx=ctx, raw_score=raw_score)
            score_rows.append(
                {
                    "score_raw": float(raw_score),
                    "score_quality": float(quality_score),
                    "metricx_clamped": bool(metricx_clamped),
                }
            )
        return score_rows

    if mode == "subprocess":
        mt_key = f"mt_{q_tag}"
        requests: list[dict[str, Any]] = []
        request_ids: list[str] = []
        for row in rows:
            request_id = f"{ctx.run_id}/subsets/subset_{ctx.subset_idx:03d}/{row['id']}/{q_tag}"
            request = QeIsolationRequest(
                id=request_id,
                row_id=str(row["id"]),
                q_tag=q_tag,
                backend=backend,
                src=str(row["source"]),
                mt=str(row[mt_key]),
                run_id=ctx.run_id,
                subset_idx=ctx.subset_idx,
                phase=f"infer-{q_tag}",
            ).to_dict()
            request["runtime_config"] = {
                "qe_primary": _get_by_dotpath(ctx.cfg, "qe.primary", {}),
                "qe_scoring": _get_by_dotpath(ctx.cfg, "qe.scoring", {}),
                "data_length": _get_by_dotpath(ctx.cfg, "data.length", {}),
            }
            requests.append(request)
            request_ids.append(request_id)

        response_rows = _run_subprocess_jsonl(
            ctx=ctx,
            section="qe",
            phase=f"qe-{q_tag}",
            input_rows=requests,
        )

        by_id: dict[str, QeIsolationResponse] = {}
        for row in response_rows:
            parsed = QeIsolationResponse.from_dict(row)
            if parsed.status not in {None, "ok"}:
                raise StepSubsetError(
                    f"qe subprocess row failed for id={parsed.id}: {parsed.error}"
                )
            by_id[parsed.id] = parsed

        out_score_rows: list[dict[str, Any]] = []
        for req_id in request_ids:
            parsed = by_id.get(req_id)
            if parsed is None:
                raise StepSubsetError(f"qe subprocess missing response for id={req_id}")
            raw_score = float(parsed.score)
            if not math.isfinite(raw_score):
                raise StepSubsetError(f"qe subprocess returned non-finite score for id={req_id}")
            quality_score, metricx_clamped = _qe_quality_from_raw(ctx=ctx, raw_score=raw_score)
            out_score_rows.append(
                {
                    "score_raw": raw_score,
                    "score_quality": float(quality_score),
                    "metricx_clamped": bool(metricx_clamped),
                }
            )
        return out_score_rows

    raise StepSubsetError(f"Unsupported qe runtime mode: {mode}")


def run_infer_q1(
    *,
    config_path: str = "configs/scp_stage4.yaml",
    overrides: list[str] | None = None,
    run_id_override: str | None = None,
    subset_idx: int = 0,
    subset_size_override: int | None = None,
    use_prepared_data: bool = True,
    use_sampled_data: bool = True,
) -> dict[str, Any]:
    ctx = _build_context(
        config_path=config_path,
        overrides=overrides,
        run_id_override=run_id_override,
        subset_idx=subset_idx,
    )
    input_rows = _materialize_input_rows(
        ctx,
        subset_size_override=subset_size_override,
        use_prepared_data=use_prepared_data,
        use_sampled_data=use_sampled_data,
    )

    q1_rows = _generate_mt_rows(ctx=ctx, rows=input_rows, q_tag="q1")
    qe_scores = _score_mt_rows(ctx=ctx, rows=q1_rows, q_tag="q1")
    for row, score in zip(q1_rows, qe_scores):
        row["qe_q1"] = float(score["score_quality"])
        row["qe_raw_q1"] = float(score["score_raw"])
        row["metricx_q1_clamped"] = bool(score["metricx_clamped"])

    q1_rows = _write_artifact(ctx.subset_root / "q1.jsonl", q1_rows, "q1")
    validate_row_id_preservation(input_rows, q1_rows, base_name="input", candidate_name="q1")

    ctx.logger.log_event(
        context=_context_for_phase(ctx, "infer-q1"),
        event_type="phase_completed",
        status="ok",
        artifact_path=f"subsets/subset_{ctx.subset_idx:03d}/q1.jsonl",
    )
    ctx.logger.log_metrics(
        context=_context_for_phase(ctx, "infer-q1"),
        metrics={"subset/input_rows": len(input_rows), "subset/q1_rows": len(q1_rows)},
        metric_group="subset",
    )
    _touch_failure_layout(ctx)
    return {
        "run_id": ctx.run_id,
        "subset_idx": ctx.subset_idx,
        "run_root": str(ctx.run_root),
        "input_rows": len(input_rows),
        "q1_rows": len(q1_rows),
    }


def run_train_collapse_lora(
    *,
    config_path: str = "configs/scp_stage4.yaml",
    overrides: list[str] | None = None,
    run_id_override: str | None = None,
    subset_idx: int = 0,
) -> dict[str, Any]:
    ctx = _build_context(
        config_path=config_path,
        overrides=overrides,
        run_id_override=run_id_override,
        subset_idx=subset_idx,
    )
    q1_rows = _read_artifact(ctx.subset_root / "q1.jsonl", "q1")

    adapter_path = ctx.subset_root / "collapse_adapter"
    mode = _training_runtime_mode(ctx)
    if mode == "mock":
        status_rows = [
            {
                "status": "ok",
                "adapter_path": str(adapter_path),
                "trained_rows": len(q1_rows),
                "backend": "mock",
            }
        ]
    elif mode == "subprocess":
        requests = [
            {
                "id": row["id"],
                "run_id": ctx.run_id,
                "subset_idx": ctx.subset_idx,
                "phase": "train-collapse-lora",
                "source": row["source"],
                "target": row["mt_q1"],
                "metadata": row.get("metadata", {}),
                "adapter_path": str(adapter_path),
                "training_config": _get_by_dotpath(ctx.cfg, "training.collapse_lora", {}),
                "model": _get_by_dotpath(ctx.cfg, "model", {}),
                "base_checkpoint": _latest_checkpoint_ref(ctx),
                "logging_config": _get_by_dotpath(ctx.cfg, "logging", {}),
            }
            for row in q1_rows
        ]
        status_rows = _run_training_subprocess_jsonl(
            ctx=ctx,
            command_key="collapse_command",
            phase="train-collapse-lora",
            input_rows=requests,
        )
        _validate_status_rows(status_rows, phase="train-collapse-lora")
    else:
        raise StepSubsetError(f"Unsupported training runtime mode: {mode}")

    adapter_path.mkdir(parents=True, exist_ok=True)
    state = {
        "status": "ok",
        "adapter_path": str(adapter_path),
        "run_id": ctx.run_id,
        "subset_idx": ctx.subset_idx,
        "trained_rows": len(q1_rows),
        "runtime_mode": mode,
        "status_rows": status_rows,
    }
    _write_json_file(_collapse_state_path(ctx), state)

    ctx.logger.log_event(
        context=_context_for_phase(ctx, "train-collapse-lora"),
        event_type="phase_completed",
        status="ok",
        artifact_path=f"subsets/subset_{ctx.subset_idx:03d}/collapse_adapter/collapse_state.json",
    )
    ctx.logger.log_metrics(
        context=_context_for_phase(ctx, "train-collapse-lora"),
        metrics={"subset/collapse_train_rows": len(q1_rows)},
        metric_group="subset",
    )
    _touch_failure_layout(ctx)
    return {
        "run_id": ctx.run_id,
        "subset_idx": ctx.subset_idx,
        "run_root": str(ctx.run_root),
        "collapse_train_rows": len(q1_rows),
        "adapter_path": str(adapter_path),
    }


def run_infer_q2(
    *,
    config_path: str = "configs/scp_stage4.yaml",
    overrides: list[str] | None = None,
    run_id_override: str | None = None,
    subset_idx: int = 0,
) -> dict[str, Any]:
    ctx = _build_context(
        config_path=config_path,
        overrides=overrides,
        run_id_override=run_id_override,
        subset_idx=subset_idx,
    )
    q1_rows = _read_artifact(ctx.subset_root / "q1.jsonl", "q1")
    _collapse_adapter_ref(ctx)

    q2_rows = _generate_mt_rows(ctx=ctx, rows=q1_rows, q_tag="q2")
    qe_scores = _score_mt_rows(ctx=ctx, rows=q2_rows, q_tag="q2")
    for row, score in zip(q2_rows, qe_scores):
        row["qe_q2"] = float(score["score_quality"])
        row["qe_raw_q2"] = float(score["score_raw"])
        row["metricx_q2_clamped"] = bool(score["metricx_clamped"])

    q2_rows = _write_artifact(ctx.subset_root / "q2.jsonl", q2_rows, "q2")
    validate_row_id_preservation(q1_rows, q2_rows, base_name="q1", candidate_name="q2")

    ctx.logger.log_event(
        context=_context_for_phase(ctx, "infer-q2"),
        event_type="phase_completed",
        status="ok",
        artifact_path=f"subsets/subset_{ctx.subset_idx:03d}/q2.jsonl",
    )
    ctx.logger.log_metrics(
        context=_context_for_phase(ctx, "infer-q2"),
        metrics={"subset/q2_rows": len(q2_rows)},
        metric_group="subset",
    )
    _touch_failure_layout(ctx)
    return {
        "run_id": ctx.run_id,
        "subset_idx": ctx.subset_idx,
        "run_root": str(ctx.run_root),
        "q2_rows": len(q2_rows),
    }


def run_unload_collapse_lora(
    *,
    config_path: str = "configs/scp_stage4.yaml",
    overrides: list[str] | None = None,
    run_id_override: str | None = None,
    subset_idx: int = 0,
) -> dict[str, Any]:
    ctx = _build_context(
        config_path=config_path,
        overrides=overrides,
        run_id_override=run_id_override,
        subset_idx=subset_idx,
    )
    q2_rows = _read_artifact(ctx.subset_root / "q2.jsonl", "q2")
    collapse_adapter = _collapse_adapter_ref(ctx)

    mode = _training_runtime_mode(ctx)
    if mode == "mock":
        status_rows = [
            {
                "status": "ok",
                "adapter_path": collapse_adapter,
                "clean_base": True,
                "active_adapters": [],
                "collapse_merged": False,
                "adapter_registry_hash": hashlib.sha256(
                    f"{ctx.run_id}:{ctx.subset_idx}:mock".encode("utf-8")
                ).hexdigest(),
                "verified_adapter_path": collapse_adapter,
                "backend": "mock",
            }
        ]
    elif mode == "subprocess":
        status_rows = _run_training_subprocess_jsonl(
            ctx=ctx,
            command_key="unload_command",
            phase="unload-collapse-lora",
            input_rows=[
                {
                    "run_id": ctx.run_id,
                    "subset_idx": ctx.subset_idx,
                    "phase": "unload-collapse-lora",
                    "adapter_path": collapse_adapter,
                    "base_checkpoint": _latest_checkpoint_ref(ctx),
                }
            ],
        )
        _validate_status_rows(status_rows, phase="unload-collapse-lora")
    else:
        raise StepSubsetError(f"Unsupported training runtime mode: {mode}")

    evidence = _normalize_clean_base_evidence(
        status_row=status_rows[0],
        collapse_adapter=collapse_adapter,
        strict=(mode == "subprocess"),
    )
    clean_state = {
        "status": "ok",
        "run_id": ctx.run_id,
        "subset_idx": ctx.subset_idx,
        "verified_rows": len(q2_rows),
        "collapse_adapter": collapse_adapter,
        "clean_base": evidence["clean_base"],
        "active_adapters": evidence["active_adapters"],
        "collapse_merged": evidence["collapse_merged"],
        "adapter_registry_hash": evidence["adapter_registry_hash"],
        "verified_adapter_path": evidence["verified_adapter_path"],
        "runtime_mode": mode,
        "status_rows": status_rows,
    }
    _write_json_file(_clean_base_state_path(ctx), clean_state)

    ctx.logger.log_event(
        context=_context_for_phase(ctx, "unload-collapse-lora"),
        event_type="phase_completed",
        status="ok",
        artifact_path=f"subsets/subset_{ctx.subset_idx:03d}/clean_base.json",
    )
    ctx.logger.log_metrics(
        context=_context_for_phase(ctx, "unload-collapse-lora"),
        metrics={"subset/clean_base_verified": 1},
        metric_group="subset",
    )
    _touch_failure_layout(ctx)
    return {
        "run_id": ctx.run_id,
        "subset_idx": ctx.subset_idx,
        "run_root": str(ctx.run_root),
        "clean_base": True,
    }


def _select_fragile(scored_rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    threshold = float(
        _get_by_dotpath(cfg, "qe.scoring.selection.default_rule.require_score_s_gte", 0.0)
    )
    top_fraction = float(
        _get_by_dotpath(cfg, "qe.scoring.selection.default_rule.top_fraction", 0.1)
    )

    eligible = [dict(row) for row in scored_rows if float(row["score_s"]) >= threshold]
    eligible_sorted = sorted(eligible, key=lambda row: (-float(row["score_s"]), str(row["id"])))

    keep = max(1, int(len(scored_rows) * top_fraction + 0.999999))
    ranked = eligible_sorted[: min(keep, len(eligible_sorted))]
    rank_by_id = {row["id"]: idx for idx, row in enumerate(ranked, start=1)}

    selected: list[dict[str, Any]] = []
    for row in scored_rows:
        row_id = str(row["id"])
        if row_id not in rank_by_id:
            continue
        out = dict(row)
        out["selection_rank"] = rank_by_id[row_id]
        out["selection_rule"] = "default_rule:score_s_gte_and_top_fraction"
        selected.append(out)
    return selected


def run_score(
    *,
    config_path: str = "configs/scp_stage4.yaml",
    overrides: list[str] | None = None,
    run_id_override: str | None = None,
    subset_idx: int = 0,
) -> dict[str, Any]:
    ctx = _build_context(
        config_path=config_path,
        overrides=overrides,
        run_id_override=run_id_override,
        subset_idx=subset_idx,
    )
    q2_rows = _read_artifact(ctx.subset_root / "q2.jsonl", "q2")
    epsilon = float(_get_by_dotpath(ctx.cfg, "qe.epsilon", 1e-6))
    if epsilon <= 0:
        raise StepSubsetError("qe.epsilon must be > 0")
    alpha = float(_get_by_dotpath(ctx.cfg, "qe.scoring.weighted_score.alpha", 0.3))
    beta = float(_get_by_dotpath(ctx.cfg, "qe.scoring.weighted_score.beta", 1.0 - alpha))
    weighted_enabled = bool(_get_by_dotpath(ctx.cfg, "qe.scoring.weighted_score.enabled", True))

    difficulty_terms: list[float] = []
    collapse_terms: list[float] = []
    scored_rows: list[dict[str, Any]] = []
    for row in q2_rows:
        q1_quality = float(row["qe_q1"])
        q2_quality = float(row["qe_q2"])
        delta_qe = q2_quality - q1_quality
        collapse_term = max((q1_quality - q2_quality) / max(q1_quality + epsilon, epsilon), 0.0)
        difficulty_term = -math.log(max(q1_quality + epsilon, epsilon))
        out = dict(row)
        out["qe_q1"] = q1_quality
        out["qe_q2"] = q2_quality
        out["delta_qe"] = round(delta_qe, 6)
        out["collapse_term"] = round(collapse_term, 6)
        out["difficulty_term"] = round(difficulty_term, 6)
        difficulty_terms.append(difficulty_term)
        collapse_terms.append(collapse_term)
        scored_rows.append(out)

    difficulty_z = _zscore(difficulty_terms)
    collapse_z = _zscore(collapse_terms)
    for idx, row in enumerate(scored_rows):
        row["difficulty_z"] = round(difficulty_z[idx], 6)
        row["collapse_z"] = round(collapse_z[idx], 6)
        if weighted_enabled:
            score_s = alpha * difficulty_z[idx] + beta * collapse_z[idx]
        else:
            score_s = collapse_terms[idx]
        row["score_s"] = round(score_s, 6)

    selected_rows = _select_fragile(scored_rows, ctx.cfg)

    scored_rows = _write_artifact(ctx.subset_root / "scored.jsonl", scored_rows, "scored")
    selected_rows = _write_artifact(ctx.subset_root / "selected.jsonl", selected_rows, "selected")

    validate_row_id_preservation(q2_rows, scored_rows, base_name="q2", candidate_name="scored")
    validate_row_id_preservation(
        scored_rows,
        selected_rows,
        allow_subset=True,
        base_name="scored",
        candidate_name="selected",
    )

    ctx.logger.log_event(
        context=_context_for_phase(ctx, "score"),
        event_type="phase_completed",
        status="ok",
        artifact_path=f"subsets/subset_{ctx.subset_idx:03d}/scored.jsonl",
        metrics={"scored_rows": len(scored_rows), "selected_rows": len(selected_rows)},
    )
    ctx.logger.log_metrics(
        context=_context_for_phase(ctx, "score"),
        metrics={"subset/scored_rows": len(scored_rows), "subset/selected_rows": len(selected_rows)},
        metric_group="subset",
    )
    _touch_failure_layout(ctx)
    return {
        "run_id": ctx.run_id,
        "subset_idx": ctx.subset_idx,
        "run_root": str(ctx.run_root),
        "scored_rows": len(scored_rows),
        "selected_rows": len(selected_rows),
    }


def _prompt_hash(ctx: PipelineContext) -> str:
    prompt_cfg = _get_by_dotpath(ctx.cfg, "prompts", {})
    if not isinstance(prompt_cfg, Mapping):
        prompt_cfg = {}
    canonical = json.dumps(prompt_cfg, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _allowed_api_statuses(ctx: PipelineContext) -> set[str]:
    allowed_raw = _get_by_dotpath(
        ctx.cfg,
        "external_api.output_status.allowed",
        ["ok", "skipped", "filtered", "needs_review", "failed"],
    )
    if not isinstance(allowed_raw, list):
        raise StepSubsetError("external_api.output_status.allowed must be a list")
    allowed = {str(value) for value in allowed_raw}
    required = {"ok", "skipped", "filtered", "needs_review", "failed"}
    if not required.issubset(allowed):
        raise StepSubsetError(
            "external_api.output_status.allowed must include: ok, skipped, filtered, needs_review, failed"
        )
    return allowed


def _normalize_api_response_row(
    *,
    ctx: PipelineContext,
    request_row: Mapping[str, Any],
    runtime_resp: Mapping[str, Any] | None,
) -> dict[str, Any]:
    req_id = str(request_row["request_id"])
    allowed_status = _allowed_api_statuses(ctx)
    provider = str(request_row["provider"])
    model = str(request_row["model"])
    prompt_version = str(request_row["prompt_version"])
    prompt_hash = str(request_row["prompt_hash"])

    runtime_resp = runtime_resp or {}
    status = str(runtime_resp.get("status", "ok"))
    if status not in allowed_status:
        raise StepSubsetError(f"external_api status={status!r} is not allowed by config")

    teacher_label = runtime_resp.get("teacher_label")
    if not isinstance(teacher_label, str) or not teacher_label.strip():
        teacher_label = "minor_edit" if status == "ok" else "invalid"

    reason = runtime_resp.get("reason")
    if reason is None:
        if status != "ok":
            reason = runtime_resp.get("error") or f"status={status}"
    if reason is not None and not isinstance(reason, str):
        reason = str(reason)

    gold_value = runtime_resp.get("gold")
    if status == "ok":
        if not isinstance(gold_value, str) or not gold_value.strip():
            raise StepSubsetError(
                f"external_api subprocess response missing gold for request_id={req_id}"
            )
        gold = str(gold_value)
    else:
        gold = None

    usage = runtime_resp.get("usage", {})
    if not isinstance(usage, Mapping):
        usage = {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)

    cost = runtime_resp.get("cost", {})
    if not isinstance(cost, Mapping):
        cost = {}
    currency = str(cost.get("currency", "USD"))
    estimated_cost = float(cost.get("estimated", 0.0) or 0.0)

    latency_ms = float(runtime_resp.get("latency_ms", 0.0) or 0.0)
    attempt = int(runtime_resp.get("attempt", 1) or 1)
    error = runtime_resp.get("error")
    if error is not None and not isinstance(error, str):
        error = str(error)

    response = {
        "id": request_row["id"],
        "row_id": request_row["row_id"],
        "dataset": request_row["dataset"],
        "source": request_row["source"],
        "metadata": request_row["metadata"],
        "request_id": req_id,
        "run_id": request_row["run_id"],
        "subset_idx": request_row["subset_idx"],
        "provider": provider,
        "model": model,
        "status": status,
        "teacher_label": str(teacher_label),
        "student": request_row["student"],
        "gold": gold,
        "reason": reason,
        "prompt_version": prompt_version,
        "prompt_hash": prompt_hash,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
        "cost": {
            "currency": currency,
            "estimated": estimated_cost,
        },
        "latency_ms": latency_ms,
        "attempt": attempt,
        "error": error,
        "config_hash": request_row["config_hash"],
    }
    return response


def _derive_error_type_from_api_row(row: Mapping[str, Any]) -> str:
    status = str(row.get("status", "failed"))
    if status == "ok":
        return "none"
    if status in {"skipped", "filtered", "needs_review"}:
        return status
    reason = str(row.get("reason", "") or "").lower()
    error = str(row.get("error", "") or "").lower()
    if "timeout" in reason or "timeout" in error:
        return "timeout"
    if error.strip():
        return "runtime_error"
    return "failed"


def _build_preference_pairs(
    *,
    api_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for row in api_rows:
        pairs.append(
            {
                "id": row["id"],
                "row_id": row["row_id"],
                "request_id": row["request_id"],
                "run_id": row["run_id"],
                "subset_idx": row["subset_idx"],
                "dataset": row["dataset"],
                "source": row["source"],
                "metadata": row["metadata"],
                "student": row["student"],
                "gold": row.get("gold"),
                "status": row["status"],
                "error_type": _derive_error_type_from_api_row(row),
                "teacher_label": row["teacher_label"],
                "reason": row.get("reason"),
                "error": row.get("error"),
                "provider": row.get("provider"),
                "model": row.get("model"),
                "prompt_version": row.get("prompt_version"),
                "prompt_hash": row.get("prompt_hash"),
                "usage": row.get("usage"),
                "cost": row.get("cost"),
                "latency_ms": row.get("latency_ms"),
                "attempt": row.get("attempt"),
                "config_hash": row.get("config_hash"),
            }
        )
    return pairs


def _upsert_run_level_preference_pairs(
    *,
    ctx: PipelineContext,
    subset_pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    run_path = ctx.run_root / "preference_pairs.jsonl"
    existing_rows: list[dict[str, Any]] = []
    if run_path.exists():
        existing_rows = _as_rows(read_jsonl(run_path))
        existing_rows = validate_artifact_rows(existing_rows, "preference_pairs")

    subset_indices = {int(row["subset_idx"]) for row in subset_pairs}
    carried = [
        dict(row) for row in existing_rows if int(row["subset_idx"]) not in subset_indices
    ]
    merged = carried + [dict(row) for row in subset_pairs]

    dedup_by_request_id: dict[str, dict[str, Any]] = {}
    for row in merged:
        dedup_by_request_id[str(row["request_id"])] = row
    merged_rows = list(dedup_by_request_id.values())
    merged_rows.sort(
        key=lambda row: (
            int(row["subset_idx"]),
            str(row["id"]),
            str(row["request_id"]),
        )
    )
    write_jsonl(run_path, validate_artifact_rows(merged_rows, "preference_pairs"), ensure_ascii=False)
    return validate_artifact_rows(_as_rows(read_jsonl(run_path)), "preference_pairs")


def _build_api_requests(
    *,
    ctx: PipelineContext,
    selected_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    provider = str(_get_by_dotpath(ctx.cfg, "external_api.primary.provider", "openai"))
    model = str(_get_by_dotpath(ctx.cfg, "external_api.primary.model", "unknown"))
    prompt_version = str(_get_by_dotpath(ctx.cfg, "prompts.version", "teacher_correction_v1"))
    prompt_hash = _prompt_hash(ctx)

    requests: list[dict[str, Any]] = []
    for row in selected_rows:
        request_id = f"{ctx.run_id}/subsets/subset_{ctx.subset_idx:03d}/{row['id']}/api"
        requests.append(
            {
                "id": row["id"],
                "row_id": row["id"],
                "dataset": row["dataset"],
                "source": row["source"],
                "metadata": row["metadata"],
                "request_id": request_id,
                "run_id": ctx.run_id,
                "subset_idx": ctx.subset_idx,
                "student": row["mt_q1"],
                "selection": {
                    "score_s": float(row["score_s"]),
                    "qe_q1": float(row["qe_q1"]),
                    "qe_q2": float(row["qe_q2"]),
                    "delta_qe": float(row.get("delta_qe", 0.0)),
                    "collapse_term": float(row.get("collapse_term", 0.0)),
                },
                "prompt_version": prompt_version,
                "prompt_hash": prompt_hash,
                "provider": provider,
                "model": model,
                "status": "ok",
                "config_hash": ctx.cfg_hash,
            }
        )
    return requests


def _mock_api_responses(
    *,
    ctx: PipelineContext,
    requests: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    for row in requests:
        runtime_resp = {
            "request_id": row["request_id"],
            "status": "ok",
            "teacher_label": "minor_edit",
            "gold": f"KO_GOLD::{row['id']}",
            "usage": {
                "input_tokens": 64,
                "output_tokens": 48,
                "total_tokens": 112,
            },
            "cost": {
                "currency": "USD",
                "estimated": 0.0,
            },
            "latency_ms": 1.0,
            "attempt": 1,
            "error": None,
        }
        responses.append(
            _normalize_api_response_row(
                ctx=ctx,
                request_row=row,
                runtime_resp=runtime_resp,
            )
        )
    return responses


def _subprocess_api_responses(
    *,
    ctx: PipelineContext,
    requests: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    runtime_requests = []
    for row in requests:
        payload = dict(row)
        payload["runtime_config"] = {
            "external_api": _get_by_dotpath(ctx.cfg, "external_api", {}),
            "prompts": _get_by_dotpath(ctx.cfg, "prompts", {}),
        }
        runtime_requests.append(payload)
    runtime_responses = _run_subprocess_jsonl(
        ctx=ctx,
        section="external_api",
        phase="call-api",
        input_rows=runtime_requests,
    )

    by_request_id: dict[str, dict[str, Any]] = {}
    for resp in runtime_responses:
        req_id = resp.get("request_id")
        if not isinstance(req_id, str) or not req_id:
            raise StepSubsetError("external_api subprocess response missing request_id")
        by_request_id[req_id] = resp

    responses: list[dict[str, Any]] = []
    for req in requests:
        req_id = str(req["request_id"])
        runtime_resp = by_request_id.get(req_id)
        if runtime_resp is None:
            raise StepSubsetError(
                f"external_api subprocess missing response for request_id={req_id}"
            )
        responses.append(
            _normalize_api_response_row(
                ctx=ctx,
                request_row=req,
                runtime_resp=runtime_resp,
            )
        )

    return responses


def run_call_api(
    *,
    config_path: str = "configs/scp_stage4.yaml",
    overrides: list[str] | None = None,
    run_id_override: str | None = None,
    subset_idx: int = 0,
) -> dict[str, Any]:
    ctx = _build_context(
        config_path=config_path,
        overrides=overrides,
        run_id_override=run_id_override,
        subset_idx=subset_idx,
    )
    selected_rows = _read_artifact(ctx.subset_root / "selected.jsonl", "selected")
    _assert_clean_base(ctx)

    requests = _build_api_requests(ctx=ctx, selected_rows=selected_rows)
    requests = _write_artifact(ctx.subset_root / "api_requests.jsonl", requests, "api_requests")

    mode = _runtime_mode(ctx, "external_api")
    if mode == "mock":
        responses = _mock_api_responses(ctx=ctx, requests=requests)
    elif mode == "subprocess":
        responses = _subprocess_api_responses(ctx=ctx, requests=requests)
    else:
        raise StepSubsetError(f"Unsupported external_api runtime mode: {mode}")

    responses = _write_artifact(ctx.subset_root / "api.jsonl", responses, "api")
    preference_pairs = _build_preference_pairs(api_rows=responses)
    preference_pairs = _write_artifact(
        ctx.subset_root / "preference_pairs.jsonl",
        preference_pairs,
        "preference_pairs",
    )
    run_level_preference_pairs = _upsert_run_level_preference_pairs(
        ctx=ctx,
        subset_pairs=preference_pairs,
    )

    validate_row_id_preservation(
        selected_rows,
        requests,
        allow_subset=True,
        base_name="selected",
        candidate_name="api_requests",
    )
    validate_row_id_preservation(
        requests,
        responses,
        allow_subset=True,
        base_name="api_requests",
        candidate_name="api",
    )
    validate_row_id_preservation(
        responses,
        preference_pairs,
        allow_subset=True,
        base_name="api",
        candidate_name="preference_pairs",
    )

    ctx.logger.log_event(
        context=_context_for_phase(ctx, "call-api"),
        event_type="phase_completed",
        status="ok",
        artifact_path=f"subsets/subset_{ctx.subset_idx:03d}/api.jsonl",
        metrics={"api_requests": len(requests), "api_rows": len(responses)},
    )
    ctx.logger.log_metrics(
        context=_context_for_phase(ctx, "call-api"),
        metrics={
            "subset/api_ok_rows": len([row for row in responses if row["status"] == "ok"]),
            "subset/api_failed_rows": len([row for row in responses if row["status"] != "ok"]),
        },
        metric_group="subset",
    )
    _touch_failure_layout(ctx)
    return {
        "run_id": ctx.run_id,
        "subset_idx": ctx.subset_idx,
        "run_root": str(ctx.run_root),
        "api_requests": len(requests),
        "api_rows": len(responses),
        "preference_pairs": len(preference_pairs),
        "preference_pairs_run_total": len(run_level_preference_pairs),
    }


def run_update_base(
    *,
    config_path: str = "configs/scp_stage4.yaml",
    overrides: list[str] | None = None,
    run_id_override: str | None = None,
    subset_idx: int = 0,
) -> dict[str, Any]:
    ctx = _build_context(
        config_path=config_path,
        overrides=overrides,
        run_id_override=run_id_override,
        subset_idx=subset_idx,
    )
    api_rows = _read_artifact(ctx.subset_root / "api.jsonl", "api")
    _assert_clean_base(ctx)

    train_rows: list[dict[str, Any]] = []
    for row in api_rows:
        if row["status"] != "ok":
            continue
        train_rows.append(
            {
                "id": row["id"],
                "dataset": row["dataset"],
                "source": row["source"],
                "gold": row["gold"],
                "metadata": row["metadata"],
            }
        )

    train_path = ctx.subset_root / "train_final" / "train_rows.jsonl"
    train_rows = _write_artifact(train_path, train_rows, "train")
    validate_row_id_preservation(
        api_rows,
        train_rows,
        allow_subset=True,
        base_name="api",
        candidate_name="train",
    )

    mode = _training_runtime_mode(ctx)
    train_final_dir = ctx.subset_root / "train_final"
    checkpoint_path = train_final_dir / "main_adapter"
    if mode == "mock":
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        status_rows = [
            {
                "status": "ok",
                "checkpoint_path": str(checkpoint_path),
                "trained_rows": len(train_rows),
                "backend": "mock",
            }
        ]
    elif mode == "subprocess":
        status_rows = _run_training_subprocess_jsonl(
            ctx=ctx,
            command_key="update_command",
            phase="update-base",
            input_rows=[
                {
                    "id": row["id"],
                    "run_id": ctx.run_id,
                    "subset_idx": ctx.subset_idx,
                    "phase": "update-base",
                    "source": row["source"],
                    "target": row["gold"],
                    "metadata": row.get("metadata", {}),
                    "train_artifact": str(train_path),
                    "output_dir": str(train_final_dir),
                    "training_config": _get_by_dotpath(ctx.cfg, "training.base_update", {}),
                    "model": _get_by_dotpath(ctx.cfg, "model", {}),
                    "base_checkpoint": _latest_checkpoint_ref(ctx),
                    "logging_config": _get_by_dotpath(ctx.cfg, "logging", {}),
                }
                for row in train_rows
            ],
        )
        _validate_status_rows(status_rows, phase="update-base")
        for row in status_rows:
            maybe_checkpoint = row.get("checkpoint_path")
            if isinstance(maybe_checkpoint, str) and maybe_checkpoint.strip():
                checkpoint_path = Path(maybe_checkpoint)
                break
    else:
        raise StepSubsetError(f"Unsupported training runtime mode: {mode}")

    checkpoint_state = {
        "status": "ok",
        "run_id": ctx.run_id,
        "subset_idx": ctx.subset_idx,
        "checkpoint_path": str(checkpoint_path),
        "train_rows": len(train_rows),
        "runtime_mode": mode,
        "status_rows": status_rows,
    }
    _write_json_file(train_final_dir / "checkpoint_state.json", checkpoint_state)
    _write_json_file(_latest_checkpoint_path(ctx), checkpoint_state)
    prune_stats = _prune_subset_checkpoints_if_configured(ctx)

    ctx.logger.log_event(
        context=_context_for_phase(ctx, "update-base"),
        event_type="phase_completed",
        status="ok",
        artifact_path=f"subsets/subset_{ctx.subset_idx:03d}/train_final/checkpoint_state.json",
        extras={"checkpoint_prune": prune_stats},
    )
    ctx.logger.log_metrics(
        context=_context_for_phase(ctx, "update-base"),
        metrics={
            "subset/train_rows": len(train_rows),
            "checkpoint/retained_count": int(prune_stats["preserved_subset_count"]),
            "checkpoint/retained_best_count": int(prune_stats["preserved_best_count"]),
            "checkpoint/keep_last_n": _checkpoint_retention_keep_last_n(ctx.cfg),
            "checkpoint/keep_best_n": _checkpoint_retention_keep_best_n(ctx.cfg),
            "checkpoint/pruned_subset_count": int(prune_stats["subset_count"]),
            "checkpoint/deleted_count": int(prune_stats["deleted_count"]),
            "checkpoint/freed_bytes": int(prune_stats["freed_bytes"]),
        },
        metric_group="subset",
    )
    _touch_failure_layout(ctx)
    return {
        "run_id": ctx.run_id,
        "subset_idx": ctx.subset_idx,
        "run_root": str(ctx.run_root),
        "train_rows": len(train_rows),
        "checkpoint_path": str(checkpoint_path),
    }


def run_subset(
    *,
    config_path: str = "configs/scp_stage4.yaml",
    overrides: list[str] | None = None,
    run_id_override: str | None = None,
    subset_idx: int = 0,
    subset_size_override: int | None = None,
    use_prepared_data: bool = True,
    use_sampled_data: bool = True,
    stage_completed: bool = True,
) -> dict[str, Any]:
    infer_q1 = run_infer_q1(
        config_path=config_path,
        overrides=overrides,
        run_id_override=run_id_override,
        subset_idx=subset_idx,
        subset_size_override=subset_size_override,
        use_prepared_data=use_prepared_data,
        use_sampled_data=use_sampled_data,
    )
    collapse = run_train_collapse_lora(
        config_path=config_path,
        overrides=overrides,
        run_id_override=run_id_override,
        subset_idx=subset_idx,
    )
    infer_q2 = run_infer_q2(
        config_path=config_path,
        overrides=overrides,
        run_id_override=run_id_override,
        subset_idx=subset_idx,
    )
    scored = run_score(
        config_path=config_path,
        overrides=overrides,
        run_id_override=run_id_override,
        subset_idx=subset_idx,
    )
    clean_base = run_unload_collapse_lora(
        config_path=config_path,
        overrides=overrides,
        run_id_override=run_id_override,
        subset_idx=subset_idx,
    )
    api = run_call_api(
        config_path=config_path,
        overrides=overrides,
        run_id_override=run_id_override,
        subset_idx=subset_idx,
    )
    train = run_update_base(
        config_path=config_path,
        overrides=overrides,
        run_id_override=run_id_override,
        subset_idx=subset_idx,
    )

    ctx = _build_context(
        config_path=config_path,
        overrides=overrides,
        run_id_override=run_id_override,
        subset_idx=subset_idx,
    )

    summary = {
        "run_id": ctx.run_id,
        "subset_idx": ctx.subset_idx,
        "config_hash": ctx.cfg_hash,
        "run_root": str(ctx.run_root),
        "preference_pairs_run_total": api["preference_pairs_run_total"],
        "counts": {
            "input": infer_q1["input_rows"],
            "q1": infer_q1["q1_rows"],
            "collapse_train": collapse["collapse_train_rows"],
            "q2": infer_q2["q2_rows"],
            "scored": scored["scored_rows"],
            "selected": scored["selected_rows"],
            "clean_base": 1 if clean_base["clean_base"] else 0,
            "api_requests": api["api_requests"],
            "api": api["api_rows"],
            "preference_pairs": api["preference_pairs"],
            "train": train["train_rows"],
        },
    }
    archive = _archive_subset_if_configured(
        ctx=ctx,
        stage_completed=stage_completed,
        counts=summary["counts"],
    )
    if archive is not None:
        summary["subset_archive"] = archive

    (ctx.run_root / "run_subset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _prepared_train_rows(use_sampled_data: bool) -> list[dict[str, Any]]:
    candidates = []
    if use_sampled_data:
        candidates.append(Path("artifacts/data/datapool.train.sampled.parquet"))
        candidates.append(Path("artifacts/data/datapool.train.sampled.jsonl"))
    candidates.append(Path("artifacts/data/datapool.train.parquet"))
    candidates.append(Path("artifacts/data/datapool.train.jsonl"))
    for path in candidates:
        if path.exists():
            rows = _load_prepared_rows(path)
            if rows:
                return rows
    return []


def _subset_size_for_rows(
    rows: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    subset_size_override: int | None,
) -> int:
    if subset_size_override is not None:
        return max(1, int(subset_size_override))
    configured = _get_by_dotpath(cfg, "data.subset_size")
    if configured is not None:
        return max(1, int(configured))
    strategy = str(_get_by_dotpath(cfg, "pipeline.subset.strategy", "fraction"))
    if strategy == "fixed_size":
        return max(1, int(_get_by_dotpath(cfg, "pipeline.subset.fixed_size", 1)))
    fraction = float(_get_by_dotpath(cfg, "pipeline.subset.fraction", 0.02))
    min_size = int(_get_by_dotpath(cfg, "pipeline.subset.min_size", 32))
    size = max(min_size, int(len(rows) * fraction + 0.999999))
    max_size = _get_by_dotpath(cfg, "pipeline.subset.max_size")
    if max_size is not None:
        size = min(size, int(max_size))
    return max(1, size)


def _canonical_eval_metric(metric: str) -> str:
    text = metric.strip()
    lowered = text.lower()
    if lowered == "metricx24_ref":
        return "metricx24_ref"
    if lowered == "bleu":
        return "BLEU"
    if lowered == "chrf":
        return "chrF"
    raise StepSubsetError(
        f"Unsupported pipeline.eval_after_subset metric={metric!r}; "
        "allowed: metricx24_ref, BLEU, chrF"
    )


def _resolve_eval_metrics(eval_cfg: Mapping[str, Any]) -> list[str]:
    raw_metrics = eval_cfg.get("metrics", ["metricx24_ref", "BLEU", "chrF"])
    if not isinstance(raw_metrics, list) or not raw_metrics:
        raise StepSubsetError("pipeline.eval_after_subset.metrics must be a non-empty list")
    resolved: list[str] = []
    seen: set[str] = set()
    for raw in raw_metrics:
        canonical = _canonical_eval_metric(str(raw))
        if canonical in seen:
            continue
        seen.add(canonical)
        resolved.append(canonical)
    return resolved


def _stable_unit_interval(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 1_000_000) / 1_000_000.0


def _mock_bleu_score(mt: str, ref: str) -> float:
    mt_tokens = [tok for tok in mt.split() if tok]
    ref_tokens = [tok for tok in ref.split() if tok]
    if not mt_tokens or not ref_tokens:
        return 0.0
    mt_set = set(mt_tokens)
    ref_set = set(ref_tokens)
    precision = len(mt_set & ref_set) / max(len(mt_set), 1)
    brevity = min(len(mt_tokens), len(ref_tokens)) / max(len(mt_tokens), len(ref_tokens), 1)
    return round(_clamp((0.8 * precision + 0.2 * brevity) * 100.0, 0.0, 100.0), 6)


def _mock_chrf_score(mt: str, ref: str) -> float:
    mt_chars = {ch for ch in mt if not ch.isspace()}
    ref_chars = {ch for ch in ref if not ch.isspace()}
    if not mt_chars or not ref_chars:
        return 0.0
    precision = len(mt_chars & ref_chars) / max(len(mt_chars), 1)
    recall = len(mt_chars & ref_chars) / max(len(ref_chars), 1)
    if precision + recall <= 0:
        return 0.0
    fscore = 2 * precision * recall / (precision + recall)
    return round(_clamp(fscore * 100.0, 0.0, 100.0), 6)


def _load_ood_eval_rows(
    *,
    ctx: PipelineContext,
    dataset_name: str,
    source_column: str,
    reference_column: str,
) -> list[dict[str, Any]]:
    candidates: list[Path] = []
    dataset_token = dataset_name.strip()
    if not dataset_token:
        raise StepSubsetError("pipeline.eval_after_subset.dataset must be a non-empty string")
    dataset_path = Path(dataset_token)
    if dataset_path.suffix in {".jsonl", ".parquet"}:
        candidates.append(dataset_path)
    else:
        candidates.append(Path("artifacts/data") / f"{dataset_token}.jsonl")
        candidates.append(Path("artifacts/data") / f"{dataset_token}.parquet")

    loaded_rows: list[dict[str, Any]] = []
    used_path: Path | None = None
    for candidate in candidates:
        if candidate.exists():
            try:
                if candidate.suffix.lower() == ".parquet":
                    rows = _iter_parquet_mapping_rows(candidate)
                else:
                    rows = _as_rows(read_jsonl(candidate))
            except Exception as exc:
                raise StepSubsetError(f"failed to load OOD eval rows from {candidate}: {exc}") from exc
            if rows:
                loaded_rows = rows
                used_path = candidate
                break

    if not loaded_rows:
        # Fallback: load directly from data.ood_test CSV source when bundle restore
        # does not include ood_test.jsonl.
        ood_cfg = _get_by_dotpath(ctx.cfg, "data.ood_test", {})
        if not isinstance(ood_cfg, Mapping) or not bool(ood_cfg.get("enabled", False)):
            searched = ", ".join(str(path) for path in candidates)
            raise StepSubsetError(
                "OOD eval dataset not found or empty. "
                f"dataset={dataset_name!r} searched=[{searched}]"
            )

        csv_path = Path(str(ood_cfg.get("path", "")))
        if not csv_path.exists():
            searched = ", ".join(str(path) for path in candidates)
            raise StepSubsetError(
                "OOD eval dataset not found or empty. "
                f"dataset={dataset_name!r} searched=[{searched}] "
                f"and csv_path_missing={csv_path}"
            )

        csv_source_col = str(ood_cfg.get("source_column", source_column or "Source_En"))
        csv_target_col = str(ood_cfg.get("target_column", reference_column or "Target_Ko"))
        csv_rows: list[dict[str, Any]] = []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                if not isinstance(row, Mapping):
                    continue
                source = str(row.get(csv_source_col, "")).strip()
                target = str(row.get(csv_target_col, "")).strip()
                if not source:
                    continue
                csv_rows.append(
                    {
                        "id": f"ood_{idx:06d}",
                        "source": source,
                        "target": target,
                    }
                )
        loaded_rows = csv_rows
        used_path = csv_path

    ood_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(loaded_rows):
        source = str(row.get("source", row.get(source_column, ""))).strip()
        reference = str(
            row.get("target", row.get("reference", row.get(reference_column, "")))
        ).strip()
        if not source:
            continue
        row_id = str(row.get("id", f"{dataset_name}:{idx:06d}"))
        ood_rows.append(
            {
                "id": row_id,
                "row_id": row_id,
                "dataset": dataset_name,
                "source": source,
                "reference": reference,
                "metadata": row.get("metadata", {}),
            }
        )

    if not ood_rows:
        origin = str(used_path) if used_path is not None else dataset_name
        raise StepSubsetError(f"OOD eval rows are empty after source filtering: {origin}")
    return ood_rows


def _generate_ood_mt_rows(
    *,
    ctx: PipelineContext,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    mode = _runtime_mode(ctx, "inference")
    if mode == "mock":
        out_rows: list[dict[str, Any]] = []
        for row in rows:
            out = dict(row)
            out["mt"] = f"KO_OOD::{row['id']}"
            out_rows.append(out)
        return out_rows

    if mode == "subprocess":
        base_checkpoint = _latest_checkpoint_ref(ctx)
        requests = [
            {
                "id": f"{ctx.run_id}/subsets/subset_{ctx.subset_idx:03d}/{row['id']}/ood",
                "run_id": ctx.run_id,
                "subset_idx": ctx.subset_idx,
                "row_id": row["row_id"],
                "q_tag": "ood",
                "source": row["source"],
                "metadata": row.get("metadata", {}),
                "base_checkpoint": base_checkpoint,
                "decoding": _get_by_dotpath(ctx.cfg, "inference.q1", {}),
                "runtime_config": {
                    "model": _get_by_dotpath(ctx.cfg, "model", {}),
                    "inference": _get_by_dotpath(ctx.cfg, "inference", {}),
                    "data_length": _get_by_dotpath(ctx.cfg, "data.length", {}),
                },
            }
            for row in rows
        ]
        response_rows = _run_subprocess_jsonl(
            ctx=ctx,
            section="inference",
            phase="infer-ood",
            input_rows=requests,
        )
        by_id: dict[str, dict[str, Any]] = {}
        for resp in response_rows:
            resp_id = resp.get("id")
            mt = resp.get("mt")
            status = resp.get("status", "ok")
            if not isinstance(resp_id, str) or not resp_id:
                raise StepSubsetError("inference subprocess response missing id for eval-ood")
            if status != "ok":
                raise StepSubsetError(
                    f"inference subprocess row failed for eval-ood id={resp_id}: {resp.get('error')}"
                )
            if not isinstance(mt, str) or not mt.strip():
                raise StepSubsetError(f"inference subprocess response missing mt for eval-ood id={resp_id}")
            by_id[resp_id] = resp

        out_rows: list[dict[str, Any]] = []
        for req, row in zip(requests, rows):
            resp = by_id.get(str(req["id"]))
            if resp is None:
                raise StepSubsetError(
                    f"inference subprocess missing eval-ood response for id={req['id']}"
                )
            out = dict(row)
            out["mt"] = str(resp["mt"])
            out_rows.append(out)
        return out_rows

    raise StepSubsetError(f"Unsupported inference runtime mode for eval-ood: {mode}")


def _score_ood_metric_rows(
    *,
    ctx: PipelineContext,
    rows: Sequence[Mapping[str, Any]],
    metric_name: str,
    metric_settings: Mapping[str, Any],
) -> list[float]:
    mode = _runtime_mode(ctx, "qe")
    if mode == "mock":
        scores: list[float] = []
        for row in rows:
            source = str(row["source"])
            mt = str(row["mt"])
            reference = str(row.get("reference", ""))
            if metric_name == "metricx24_ref":
                base = _stable_unit_interval(f"{row['id']}::{source}::{mt}::{reference}")
                scores.append(round(4.0 + base * 16.0, 6))
            elif metric_name == "BLEU":
                scores.append(_mock_bleu_score(mt, reference))
            elif metric_name == "chrF":
                scores.append(_mock_chrf_score(mt, reference))
            else:
                raise StepSubsetError(f"Unsupported eval metric in mock mode: {metric_name}")
        return scores

    if mode == "subprocess":
        backend = metric_name
        requests: list[dict[str, Any]] = []
        request_ids: list[str] = []
        for row in rows:
            req_id = (
                f"{ctx.run_id}/subsets/subset_{ctx.subset_idx:03d}/{row['id']}/eval-ood/{backend}"
            )
            request = QeIsolationRequest(
                id=req_id,
                row_id=str(row["row_id"]),
                q_tag="ood",
                backend=backend,
                src=str(row["source"]),
                mt=str(row["mt"]),
                run_id=ctx.run_id,
                subset_idx=ctx.subset_idx,
                phase="eval-ood",
                ref=str(row.get("reference", "")),
            ).to_dict()
            request["runtime_config"] = {
                "qe_primary": _get_by_dotpath(ctx.cfg, "qe.primary", {}),
                "qe_scoring": _get_by_dotpath(ctx.cfg, "qe.scoring", {}),
                "metric_settings": metric_settings,
            }
            requests.append(request)
            request_ids.append(req_id)

        response_rows = _run_subprocess_jsonl(
            ctx=ctx,
            section="qe",
            phase="eval-ood",
            input_rows=requests,
        )
        by_id: dict[str, QeIsolationResponse] = {}
        for row in response_rows:
            parsed = QeIsolationResponse.from_dict(row)
            if parsed.status not in {None, "ok"}:
                raise StepSubsetError(
                    f"qe subprocess row failed for eval-ood id={parsed.id}: {parsed.error}"
                )
            by_id[parsed.id] = parsed

        scores: list[float] = []
        for req_id in request_ids:
            parsed = by_id.get(req_id)
            if parsed is None:
                raise StepSubsetError(f"qe subprocess missing eval-ood response for id={req_id}")
            score = float(parsed.score)
            if not math.isfinite(score):
                raise StepSubsetError(
                    f"qe subprocess returned non-finite eval-ood score for id={req_id}"
                )
            scores.append(score)
        return scores

    raise StepSubsetError(f"Unsupported qe runtime mode for eval-ood: {mode}")


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def run_eval_ood(
    *,
    config_path: str = "configs/scp_stage4.yaml",
    overrides: list[str] | None = None,
    run_id_override: str | None = None,
    subset_idx: int = 0,
) -> dict[str, Any]:
    ctx = _build_context(
        config_path=config_path,
        overrides=overrides,
        run_id_override=run_id_override,
        subset_idx=subset_idx,
    )
    eval_cfg = _get_by_dotpath(ctx.cfg, "pipeline.eval_after_subset", {})
    if not isinstance(eval_cfg, Mapping):
        raise StepSubsetError("pipeline.eval_after_subset must be a mapping")
    if not bool(eval_cfg.get("enabled", False)):
        raise StepSubsetError("pipeline.eval_after_subset.enabled=false; eval-ood is disabled")

    dataset_name = str(eval_cfg.get("dataset", "ood_test"))
    source_column = str(eval_cfg.get("source_column", "Source_En"))
    reference_column = str(eval_cfg.get("reference_column", "Target_Ko"))
    metric_names = _resolve_eval_metrics(eval_cfg)
    metric_settings = _get_by_dotpath(eval_cfg, "metric_settings", {})
    if not isinstance(metric_settings, Mapping):
        metric_settings = {}

    ood_rows = _load_ood_eval_rows(
        ctx=ctx,
        dataset_name=dataset_name,
        source_column=source_column,
        reference_column=reference_column,
    )
    generated_rows = _generate_ood_mt_rows(ctx=ctx, rows=ood_rows)

    for row in generated_rows:
        row["eval_dataset"] = dataset_name

    metricx_raw: list[float] = []
    metricx_quality: list[float] = []
    bleu_scores: list[float] = []
    chrf_scores: list[float] = []

    for metric_name in metric_names:
        scores = _score_ood_metric_rows(
            ctx=ctx,
            rows=generated_rows,
            metric_name=metric_name,
            metric_settings=metric_settings,
        )
        if len(scores) != len(generated_rows):
            raise StepSubsetError(
                f"eval-ood score length mismatch for {metric_name}: "
                f"expected={len(generated_rows)} got={len(scores)}"
            )

        if metric_name == "metricx24_ref":
            for row, raw in zip(generated_rows, scores):
                quality, clamped = _qe_quality_from_raw(ctx=ctx, raw_score=float(raw))
                row["metricx24_ref_raw_error"] = float(raw)
                row["metricx24_ref_quality"] = float(quality)
                row["metricx24_ref_clamped"] = bool(clamped)
                metricx_raw.append(float(raw))
                metricx_quality.append(float(quality))
        elif metric_name == "BLEU":
            for row, value in zip(generated_rows, scores):
                row["bleu"] = float(value)
                bleu_scores.append(float(value))
        elif metric_name == "chrF":
            for row, value in zip(generated_rows, scores):
                row["chrf"] = float(value)
                chrf_scores.append(float(value))

    eval_root = ctx.run_root / "eval" / dataset_name
    eval_root.mkdir(parents=True, exist_ok=True)
    rows_path = eval_root / f"subset_{ctx.subset_idx:03d}.rows.jsonl"
    summary_path = eval_root / f"subset_{ctx.subset_idx:03d}.summary.json"
    history_path = eval_root / "history.jsonl"

    write_jsonl(rows_path, generated_rows, ensure_ascii=False)

    history_rows: list[dict[str, Any]] = []
    if history_path.exists():
        loaded = _as_rows(read_jsonl(history_path))
        for row in loaded:
            if int(row.get("subset_idx", -1)) != ctx.subset_idx:
                history_rows.append(row)

    previous_quality_mean = None
    if metricx_quality:
        previous_candidates = [
            row
            for row in history_rows
            if isinstance(row.get("metricx24_ref_quality_mean"), (int, float))
            and int(row.get("subset_idx", -1)) < ctx.subset_idx
        ]
        if previous_candidates:
            previous_row = max(previous_candidates, key=lambda row: int(row.get("subset_idx", -1)))
            previous_quality_mean = float(previous_row["metricx24_ref_quality_mean"])

    metricx_quality_mean = _mean(metricx_quality) if metricx_quality else None
    quality_delta = (
        metricx_quality_mean - previous_quality_mean
        if metricx_quality_mean is not None and previous_quality_mean is not None
        else 0.0
    )

    summary: dict[str, Any] = {
        "run_id": ctx.run_id,
        "subset_idx": ctx.subset_idx,
        "config_hash": ctx.cfg_hash,
        "dataset": dataset_name,
        "rows": len(generated_rows),
        "metrics": metric_names,
        "artifact_path_rows": str(rows_path),
    }
    if metricx_raw:
        summary["metricx24_ref_raw_error_mean"] = _mean(metricx_raw)
    if metricx_quality_mean is not None:
        summary["metricx24_ref_quality_mean"] = metricx_quality_mean
        summary["metricx24_ref_quality_delta_from_previous"] = quality_delta
    if bleu_scores:
        summary["bleu_mean"] = _mean(bleu_scores)
    if chrf_scores:
        summary["chrf_mean"] = _mean(chrf_scores)

    _write_json_file(summary_path, summary)
    history_rows.append(summary)
    history_rows.sort(key=lambda row: int(row.get("subset_idx", -1)))
    write_jsonl(history_path, history_rows, ensure_ascii=False)

    log_metrics: dict[str, Any] = {
        "ood/rows": len(generated_rows),
    }
    if metricx_raw:
        log_metrics["ood/metricx24_ref_raw_error_mean"] = float(_mean(metricx_raw))
    if metricx_quality_mean is not None:
        log_metrics["ood/metricx24_ref_quality_mean"] = float(metricx_quality_mean)
        log_metrics["ood/metricx24_ref_quality_delta_from_previous"] = float(quality_delta)
    if bleu_scores:
        log_metrics["ood/bleu_mean"] = float(_mean(bleu_scores))
    if chrf_scores:
        log_metrics["ood/chrf_mean"] = float(_mean(chrf_scores))

    ctx.logger.log_event(
        context=_context_for_phase(ctx, "eval-ood"),
        event_type="phase_completed",
        status="ok",
        artifact_path=str(summary_path.relative_to(ctx.run_root)),
    )
    ctx.logger.log_metrics(
        context=_context_for_phase(ctx, "eval-ood"),
        metrics=log_metrics,
        metric_group="ood_eval",
    )
    _touch_failure_layout(ctx)

    return {
        "run_id": ctx.run_id,
        "subset_idx": ctx.subset_idx,
        "run_root": str(ctx.run_root),
        "dataset": dataset_name,
        "rows": len(generated_rows),
        "summary_path": str(summary_path),
        "rows_path": str(rows_path),
        "metrics": log_metrics,
    }


def run_stage(
    *,
    config_path: str = "configs/scp_stage4.yaml",
    overrides: list[str] | None = None,
    run_id_override: str | None = None,
    subset_size_override: int | None = None,
) -> dict[str, Any]:
    ctx = _build_context(
        config_path=config_path,
        overrides=overrides,
        run_id_override=run_id_override,
        subset_idx=0,
    )
    use_sampled_data = bool(_get_by_dotpath(ctx.cfg, "pipeline.stage.use_sampled_data", False))
    train_rows = _prepared_train_rows(use_sampled_data=use_sampled_data)
    if not train_rows:
        raise StepSubsetError("No prepared train rows found; run prepare-data before run-stage")

    subset_size = _subset_size_for_rows(train_rows, ctx.cfg, subset_size_override)
    total_subsets = int((len(train_rows) + subset_size - 1) / subset_size)
    max_subsets = _get_by_dotpath(ctx.cfg, "pipeline.stage.max_subsets")
    if max_subsets is not None:
        total_subsets = min(total_subsets, int(max_subsets))
    if bool(_get_by_dotpath(ctx.cfg, "pipeline.subset.drop_last", False)):
        total_subsets = len(train_rows) // subset_size

    allow_prefetch = bool(
        _get_by_dotpath(ctx.cfg, "pipeline.execution.allow_next_subset_q1_prefetch", False)
    )
    eval_cfg = _get_by_dotpath(ctx.cfg, "pipeline.eval_after_subset", {})
    if not isinstance(eval_cfg, Mapping):
        eval_cfg = {}
    eval_enabled = bool(eval_cfg.get("enabled", False))
    eval_every_n = int(eval_cfg.get("every_n_subsets", 1) or 1)
    if eval_every_n <= 0:
        eval_every_n = 1
    eval_run_on_final = bool(eval_cfg.get("run_on_final_subset", True))

    def _prefetch_next_input(next_idx: int) -> None:
        try:
            next_ctx = _build_context(
                config_path=config_path,
                overrides=overrides,
                run_id_override=ctx.run_id,
                subset_idx=next_idx,
            )
            _materialize_input_rows(
                next_ctx,
                subset_size_override=subset_size,
                use_prepared_data=True,
                use_sampled_data=use_sampled_data,
            )
            _prefetch_log.debug("prefetch: subset_%03d input ready", next_idx)
        except Exception as exc:
            _prefetch_log.warning("prefetch: subset_%03d failed (%s) — will re-load", next_idx, exc)

    subset_summaries: list[dict[str, Any]] = []
    eval_summaries: list[dict[str, Any]] = []
    prefetch_future: Future[None] | None = None

    with ThreadPoolExecutor(max_workers=1) as pool:
        if allow_prefetch and total_subsets > 0:
            prefetch_future = pool.submit(_prefetch_next_input, 0)

        for subset_idx in range(total_subsets):
            if prefetch_future is not None:
                try:
                    prefetch_future.result()
                except Exception:
                    pass  # logged inside _prefetch_next_input; run_subset will reload if needed
                prefetch_future = None

            next_idx = subset_idx + 1
            if allow_prefetch and next_idx < total_subsets:
                prefetch_future = pool.submit(_prefetch_next_input, next_idx)

            summary = run_subset(
                config_path=config_path,
                overrides=overrides,
                run_id_override=ctx.run_id,
                subset_idx=subset_idx,
                subset_size_override=subset_size,
                use_prepared_data=True,
                use_sampled_data=use_sampled_data,
                stage_completed=False,
            )
            should_run_eval = False
            if eval_enabled:
                is_final_subset = subset_idx == (total_subsets - 1)
                is_cadence_subset = ((subset_idx + 1) % eval_every_n) == 0
                should_run_eval = is_cadence_subset or (eval_run_on_final and is_final_subset)
            if should_run_eval:
                eval_summary = run_eval_ood(
                    config_path=config_path,
                    overrides=overrides,
                    run_id_override=ctx.run_id,
                    subset_idx=subset_idx,
                )
                summary["ood_eval"] = {
                    "rows": int(eval_summary["rows"]),
                    "summary_path": str(eval_summary["summary_path"]),
                    "rows_path": str(eval_summary["rows_path"]),
                }
                eval_summaries.append(eval_summary)
            subset_summaries.append(summary)

    archived_subset_dirs_pruned = _finalize_stage_archive_cleanup(
        ctx=ctx,
        subset_indices=[int(summary["subset_idx"]) for summary in subset_summaries],
    )

    stage_summary = {
        "run_id": ctx.run_id,
        "config_hash": ctx.cfg_hash,
        "run_root": str(ctx.run_root),
        "subset_size": subset_size,
        "subsets_run": len(subset_summaries),
        "ood_evals_run": len(eval_summaries),
        "train_rows": len(train_rows),
        "archived_subset_dirs_pruned": archived_subset_dirs_pruned,
        "subsets": subset_summaries,
    }
    if eval_summaries:
        stage_summary["ood_eval_summaries"] = [
            {
                "subset_idx": int(summary["subset_idx"]),
                "rows": int(summary["rows"]),
                "dataset": str(summary["dataset"]),
                "summary_path": str(summary["summary_path"]),
            }
            for summary in eval_summaries
        ]
    (ctx.run_root / "run_stage_summary.json").write_text(
        json.dumps(stage_summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return stage_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run stepwise local subset pipeline")
    parser.add_argument(
        "command",
        choices=[
            "infer-q1",
            "train-collapse-lora",
            "infer-q2",
            "score",
            "unload-collapse-lora",
            "call-api",
            "update-base",
            "eval-ood",
            "run-subset",
            "run-stage",
        ],
    )
    parser.add_argument("--config", default="configs/scp_stage4.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--subset-idx", type=int, default=0)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--use-prepared-data", action="store_true")
    parser.add_argument("--use-full-train-data", action="store_true")
    args, overrides = parser.parse_known_args(argv)
    phase = args.command
    try:
        if args.command == "infer-q1":
            summary = run_infer_q1(
                config_path=args.config,
                overrides=overrides,
                run_id_override=args.run_id,
                subset_idx=args.subset_idx,
                subset_size_override=args.subset_size,
                use_prepared_data=args.use_prepared_data,
                use_sampled_data=not args.use_full_train_data,
            )
        elif args.command == "train-collapse-lora":
            summary = run_train_collapse_lora(
                config_path=args.config,
                overrides=overrides,
                run_id_override=args.run_id,
                subset_idx=args.subset_idx,
            )
        elif args.command == "infer-q2":
            summary = run_infer_q2(
                config_path=args.config,
                overrides=overrides,
                run_id_override=args.run_id,
                subset_idx=args.subset_idx,
            )
        elif args.command == "unload-collapse-lora":
            summary = run_unload_collapse_lora(
                config_path=args.config,
                overrides=overrides,
                run_id_override=args.run_id,
                subset_idx=args.subset_idx,
            )
        elif args.command == "score":
            summary = run_score(
                config_path=args.config,
                overrides=overrides,
                run_id_override=args.run_id,
                subset_idx=args.subset_idx,
            )
        elif args.command == "call-api":
            summary = run_call_api(
                config_path=args.config,
                overrides=overrides,
                run_id_override=args.run_id,
                subset_idx=args.subset_idx,
            )
        elif args.command == "update-base":
            summary = run_update_base(
                config_path=args.config,
                overrides=overrides,
                run_id_override=args.run_id,
                subset_idx=args.subset_idx,
            )
        elif args.command == "eval-ood":
            summary = run_eval_ood(
                config_path=args.config,
                overrides=overrides,
                run_id_override=args.run_id,
                subset_idx=args.subset_idx,
            )
        elif args.command == "run-subset":
            summary = run_subset(
                config_path=args.config,
                overrides=overrides,
                run_id_override=args.run_id,
                subset_idx=args.subset_idx,
                subset_size_override=args.subset_size,
                use_prepared_data=args.use_prepared_data,
                use_sampled_data=not args.use_full_train_data,
            )
        else:
            summary = run_stage(
                config_path=args.config,
                overrides=overrides,
                run_id_override=args.run_id,
                subset_size_override=args.subset_size,
            )
    except Exception as exc:
        _log_cli_failure(
            config_path=args.config,
            overrides=overrides,
            run_id_override=args.run_id,
            subset_idx=args.subset_idx,
            phase=phase,
            failure=exc,
        )
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
