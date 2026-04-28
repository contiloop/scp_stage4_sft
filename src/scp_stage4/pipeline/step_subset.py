"""Stepwise local subset pipeline with mock/subprocess runtime hooks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from scp_stage4.artifacts import compute_config_hash, persist_effective_config_artifacts
from scp_stage4.config.loader import compose_config
from scp_stage4.config.validator import validate_config
from scp_stage4.data import read_jsonl, validate_row_id_preservation, write_jsonl
from scp_stage4.logging import LocalJsonlLogger, RequiredLogContext
from scp_stage4.schema import QeIsolationRequest, QeIsolationResponse, validate_artifact_rows


class StepSubsetError(RuntimeError):
    """Raised when a stepwise subset contract fails."""


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

    return PipelineContext(
        cfg=cfg,
        cfg_hash=cfg_hash,
        run_id=run_id,
        subset_idx=subset_idx,
        run_root=run_root,
        subset_root=subset_root,
        logger=logger,
    )


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
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or "no output"
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
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or "no output"
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

    pool_rows: list[dict[str, Any]] = []
    if use_prepared_data:
        prepared_candidates = []
        if use_sampled_data:
            prepared_candidates.append(Path("artifacts/data/datapool.train.sampled.jsonl"))
        prepared_candidates.append(Path("artifacts/data/datapool.train.jsonl"))
        for candidate in prepared_candidates:
            if candidate.exists():
                loaded = _as_rows(read_jsonl(candidate))
                if loaded:
                    pool_rows = validate_artifact_rows(loaded, "normalized")
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
    runtime_requests = [dict(row) for row in requests]
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

    ctx.logger.log_event(
        context=_context_for_phase(ctx, "update-base"),
        event_type="phase_completed",
        status="ok",
        artifact_path=f"subsets/subset_{ctx.subset_idx:03d}/train_final/checkpoint_state.json",
    )
    ctx.logger.log_metrics(
        context=_context_for_phase(ctx, "update-base"),
        metrics={"subset/train_rows": len(train_rows)},
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
            "train": train["train_rows"],
        },
    }
    (ctx.run_root / "run_subset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _prepared_train_rows(use_sampled_data: bool) -> list[dict[str, Any]]:
    candidates = []
    if use_sampled_data:
        candidates.append(Path("artifacts/data/datapool.train.sampled.jsonl"))
    candidates.append(Path("artifacts/data/datapool.train.jsonl"))
    for path in candidates:
        if path.exists():
            rows = _as_rows(read_jsonl(path))
            if rows:
                return validate_artifact_rows(rows, "normalized")
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

    subset_summaries = []
    for subset_idx in range(total_subsets):
        summary = run_subset(
            config_path=config_path,
            overrides=overrides,
            run_id_override=ctx.run_id,
            subset_idx=subset_idx,
            subset_size_override=subset_size,
            use_prepared_data=True,
            use_sampled_data=use_sampled_data,
        )
        subset_summaries.append(summary)

    stage_summary = {
        "run_id": ctx.run_id,
        "config_hash": ctx.cfg_hash,
        "run_root": str(ctx.run_root),
        "subset_size": subset_size,
        "subsets_run": len(subset_summaries),
        "train_rows": len(train_rows),
        "subsets": subset_summaries,
    }
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
