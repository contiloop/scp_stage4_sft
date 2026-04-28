"""Stepwise local subset pipeline for deterministic mock contract validation."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from scp_stage4.artifacts import compute_config_hash, persist_effective_config_artifacts
from scp_stage4.config.loader import compose_config
from scp_stage4.config.validator import validate_config
from scp_stage4.data import read_jsonl, validate_row_id_preservation, write_jsonl
from scp_stage4.logging import LocalJsonlLogger, RequiredLogContext
from scp_stage4.schema import validate_artifact_rows


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


def _write_artifact(path: Path, rows: Sequence[Mapping[str, Any]], artifact_name: str) -> list[dict[str, Any]]:
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
            if not rows:
                continue
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

    size = max(1, min(int(subset_size), len(shuffled)))
    return shuffled[:size]


def _to_quality_score(raw_score: float, score_direction: str) -> float:
    if score_direction == "lower_is_better":
        return -raw_score
    return raw_score


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
        write_effective_config=bool(_get_by_dotpath(cfg, "logging.local.write_effective_config", True)),
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


def _materialize_input_rows(
    ctx: PipelineContext,
    *,
    subset_size_override: int | None,
    use_prepared_data: bool,
) -> list[dict[str, Any]]:
    input_path = ctx.subset_root / "input.jsonl"

    pool_rows: list[dict[str, Any]] = []
    if use_prepared_data:
        prepared_candidates = [
            Path("artifacts/data/datapool.train.sampled.jsonl"),
            Path("artifacts/data/datapool.train.jsonl"),
        ]
        for candidate in prepared_candidates:
            if candidate.exists():
                loaded = _as_rows(read_jsonl(candidate))
                if loaded:
                    pool_rows = validate_artifact_rows(loaded, "normalized")
                    break

    if not pool_rows:
        pool_rows = _load_fixture_rows()

    selected_rows = _select_subset(pool_rows, ctx.cfg, subset_size_override)
    if not selected_rows:
        raise StepSubsetError("No rows available to build subset input")

    return _write_artifact(input_path, selected_rows, "input")


def _build_q1_rows(input_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    q1_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(input_rows):
        q1_row = dict(row)
        q1_row["mt_q1"] = f"KO_Q1::{row['id']}"
        q1_row["qe_q1"] = round(0.90 - (idx % 5) * 0.07, 6)
        q1_rows.append(q1_row)
    return q1_rows


def run_infer_q1(
    *,
    config_path: str = "configs/scp_stage4.yaml",
    overrides: list[str] | None = None,
    run_id_override: str | None = None,
    subset_idx: int = 0,
    subset_size_override: int | None = None,
    use_prepared_data: bool = True,
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
    )
    q1_rows = _build_q1_rows(input_rows)
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

    score_direction = str(_get_by_dotpath(ctx.cfg, "qe.primary.score_direction", "higher_is_better"))
    if score_direction not in {"higher_is_better", "lower_is_better"}:
        raise StepSubsetError(
            "qe.primary.score_direction must be 'higher_is_better' or 'lower_is_better'"
        )

    q2_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(q1_rows):
        qe_q1 = float(row.get("qe_q1", 0.0))
        collapse_drop = round(0.03 + (idx % 4) * 0.04, 6)
        if score_direction == "lower_is_better":
            qe_q2 = round(qe_q1 + collapse_drop, 6)
        else:
            qe_q2 = round(max(0.0, qe_q1 - collapse_drop), 6)

        q2_row = dict(row)
        q2_row["mt_q2"] = f"KO_Q2::{row['id']}"
        q2_row["qe_q2"] = qe_q2
        q2_rows.append(q2_row)

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


def _select_fragile(scored_rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    threshold = float(
        _get_by_dotpath(cfg, "qe.scoring.selection.default_rule.require_score_s_gte", 0.0)
    )
    top_fraction = float(_get_by_dotpath(cfg, "qe.scoring.selection.default_rule.top_fraction", 0.1))

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

    score_direction = str(_get_by_dotpath(ctx.cfg, "qe.primary.score_direction", "higher_is_better"))
    if score_direction not in {"higher_is_better", "lower_is_better"}:
        raise StepSubsetError(
            "qe.primary.score_direction must be 'higher_is_better' or 'lower_is_better'"
        )

    scored_rows: list[dict[str, Any]] = []
    for row in q2_rows:
        q1_quality = _to_quality_score(float(row["qe_q1"]), score_direction)
        q2_quality = _to_quality_score(float(row["qe_q2"]), score_direction)
        out = dict(row)
        out["score_s"] = round(q1_quality - q2_quality, 6)
        scored_rows.append(out)

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

    requests: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    for row in selected_rows:
        request_id = (
            f"{ctx.run_id}/subsets/subset_{ctx.subset_idx:03d}/{row['id']}/api"
        )
        request_row = {
            "id": row["id"],
            "dataset": row["dataset"],
            "source": row["source"],
            "metadata": row["metadata"],
            "request_id": request_id,
            "student": row["mt_q1"],
            "status": "ok",
        }
        response_row = {
            "id": row["id"],
            "dataset": row["dataset"],
            "source": row["source"],
            "metadata": row["metadata"],
            "request_id": request_id,
            "gold": f"KO_GOLD::{row['id']}",
            "status": "ok",
        }
        requests.append(request_row)
        responses.append(response_row)

    requests = _write_artifact(ctx.subset_root / "api_requests.jsonl", requests, "api_requests")
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
        metrics={"subset/api_ok_rows": len(responses)},
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

    ctx.logger.log_event(
        context=_context_for_phase(ctx, "update-base"),
        event_type="phase_completed",
        status="ok",
        artifact_path=f"subsets/subset_{ctx.subset_idx:03d}/train_final/train_rows.jsonl",
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
    }


def run_subset(
    *,
    config_path: str = "configs/scp_stage4.yaml",
    overrides: list[str] | None = None,
    run_id_override: str | None = None,
    subset_idx: int = 0,
    subset_size_override: int | None = None,
    use_prepared_data: bool = True,
) -> dict[str, Any]:
    infer_q1 = run_infer_q1(
        config_path=config_path,
        overrides=overrides,
        run_id_override=run_id_override,
        subset_idx=subset_idx,
        subset_size_override=subset_size_override,
        use_prepared_data=use_prepared_data,
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
            "q2": infer_q2["q2_rows"],
            "scored": scored["scored_rows"],
            "selected": scored["selected_rows"],
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run stepwise local subset pipeline")
    parser.add_argument(
        "command",
        choices=["infer-q1", "infer-q2", "score", "call-api", "update-base", "run-subset"],
    )
    parser.add_argument("--config", default="configs/scp_stage4.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--subset-idx", type=int, default=0)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--use-prepared-data", action="store_true")
    args, overrides = parser.parse_known_args(argv)

    if args.command == "infer-q1":
        summary = run_infer_q1(
            config_path=args.config,
            overrides=overrides,
            run_id_override=args.run_id,
            subset_idx=args.subset_idx,
            subset_size_override=args.subset_size,
            use_prepared_data=True,
        )
    elif args.command == "infer-q2":
        summary = run_infer_q2(
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
    else:
        summary = run_subset(
            config_path=args.config,
            overrides=overrides,
            run_id_override=args.run_id,
            subset_idx=args.subset_idx,
            subset_size_override=args.subset_size,
            use_prepared_data=True,
        )

    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
