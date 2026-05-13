"""Local smoke pipeline for lightweight contract validation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from scp_stage4.artifacts import compute_config_hash, persist_effective_config_artifacts
from scp_stage4.config.loader import compose_config
from scp_stage4.config.validator import validate_config
from scp_stage4.logging import LocalJsonlLogger, RequiredLogContext
from scp_stage4.pipeline.io_utils import iter_jsonl, write_jsonl
from scp_stage4.pipeline.prompting import teacher_prompt_hash, teacher_prompt_version
from scp_stage4.schema import validate_artifact_rows


class SmokeValidationError(RuntimeError):
    """Raised when local smoke contract checks fail."""


def _get_by_dotpath(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    cursor: Any = cfg
    for part in key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _load_fixture_rows(use_prepared_data: bool = False) -> list[dict[str, Any]]:
    if use_prepared_data:
        candidates = [
            Path("artifacts/data/datapool.train.sampled.jsonl"),
            Path("artifacts/data/datapool.train.jsonl"),
            Path("tests/fixtures/datapool.train.jsonl"),
            Path("tests/fixtures/input.jsonl"),
            Path("tests/fixtures/input.happy.jsonl"),
        ]
    else:
        candidates = [
            Path("tests/fixtures/datapool.train.jsonl"),
            Path("tests/fixtures/input.jsonl"),
            Path("tests/fixtures/input.happy.jsonl"),
            Path("artifacts/data/datapool.train.sampled.jsonl"),
            Path("artifacts/data/datapool.train.jsonl"),
        ]
    for path in candidates:
        if path.exists():
            rows: list[dict[str, Any]] = []
            for row in iter_jsonl(path):
                if "id" not in row:
                    raise SmokeValidationError(f"Fixture row missing id: {path}")
                if "source" not in row:
                    raise SmokeValidationError(f"Fixture row missing source: {path}")
                rows.append(dict(row))
            if rows:
                return rows

    # Fallback fixture for isolated local harness tests.
    rows = []
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
    return rows


def _select_subset(
    rows: list[dict[str, Any]], cfg: dict[str, Any], subset_size_override: int | None
) -> list[dict[str, Any]]:
    seed = int(_get_by_dotpath(cfg, "pipeline.subset.seed", 42))
    shuffled = list(rows)
    if bool(_get_by_dotpath(cfg, "pipeline.subset.shuffle", True)):
        rng = random.Random(seed)
        rng.shuffle(shuffled)

    subset_size = subset_size_override
    if subset_size is None:
        subset_size = _get_by_dotpath(cfg, "data.subset_size")
    if subset_size is None:
        fraction = float(_get_by_dotpath(cfg, "pipeline.subset.fraction", 0.02))
        min_size = int(_get_by_dotpath(cfg, "pipeline.subset.min_size", 32))
        subset_size = max(min_size, int(len(shuffled) * fraction + 0.999999))

    subset_size = max(1, min(int(subset_size), len(shuffled)))
    return shuffled[:subset_size]


def _build_q_rows(
    input_rows: list[dict[str, Any]],
    score_direction: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    q1_rows: list[dict[str, Any]] = []
    q2_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(input_rows):
        qe_q1 = round(0.90 - (idx % 5) * 0.07, 6)
        collapse_drop = round(0.03 + (idx % 4) * 0.04, 6)
        if score_direction == "lower_is_better":
            # For error-oriented QE (lower is better), larger Q2 means collapse.
            qe_q2 = round(qe_q1 + collapse_drop, 6)
        else:
            # For quality-oriented QE (higher is better), smaller Q2 means collapse.
            qe_q2 = round(max(0.0, qe_q1 - collapse_drop), 6)

        q1_row = dict(row)
        q1_row["mt_q1"] = f"KO_Q1::{row['id']}"
        q1_row["qe_q1"] = qe_q1

        q2_row = dict(q1_row)
        q2_row["mt_q2"] = f"KO_Q2::{row['id']}"
        q2_row["qe_q2"] = qe_q2

        q1_rows.append(q1_row)
        q2_rows.append(q2_row)

    return q1_rows, q2_rows


def _to_quality_score(raw_score: float, score_direction: str) -> float:
    if score_direction == "lower_is_better":
        return -raw_score
    return raw_score


def _score_rows(
    q2_rows: list[dict[str, Any]],
    score_direction: str,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in q2_rows:
        q1_quality = _to_quality_score(float(row["qe_q1"]), score_direction)
        q2_quality = _to_quality_score(float(row["qe_q2"]), score_direction)
        score_s = round(q1_quality - q2_quality, 6)
        scored_row = dict(row)
        scored_row["score_s"] = score_s
        scored.append(scored_row)
    return scored


def _select_fragile(scored_rows: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    top_fraction = float(
        _get_by_dotpath(cfg, "qe.scoring.selection.default_rule.top_fraction", 0.1)
    )

    eligible_sorted = sorted(
        scored_rows,
        key=lambda r: (-float(r["score_s"]), str(r["id"])),
    )

    keep = max(1, int(len(scored_rows) * top_fraction + 0.999999))
    selected_ranked = eligible_sorted[: min(keep, len(eligible_sorted))]
    rank_by_id = {row["id"]: idx for idx, row in enumerate(selected_ranked, start=1)}
    selected_id_set = set(rank_by_id.keys())

    out: list[dict[str, Any]] = []
    for row in scored_rows:
        row_id = row["id"]
        if row_id not in selected_id_set:
            continue
        row_with_rank = dict(row)
        row_with_rank["selection_rank"] = rank_by_id[row_id]
        out.append(row_with_rank)
    return out


def _make_api_artifacts(
    selected_rows: list[dict[str, Any]],
    run_id: str,
    cfg: dict[str, Any],
    cfg_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    requests: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    provider = str(_get_by_dotpath(cfg, "external_api.primary.provider", "openai"))
    model = str(_get_by_dotpath(cfg, "external_api.primary.model", "unknown"))
    prompt_cfg = _get_by_dotpath(cfg, "prompts", {})
    if not isinstance(prompt_cfg, dict):
        prompt_cfg = {}
    prompt_version = teacher_prompt_version(prompt_cfg)
    prompt_hash = teacher_prompt_hash(prompt_cfg)

    for row in selected_rows:
        request_id = f"{run_id}/subsets/subset_000/{row['id']}/api"
        qe_q1 = float(row.get("qe_q1", 0.0))
        qe_q2 = float(row.get("qe_q2", 0.0))
        delta_qe = round(qe_q2 - qe_q1, 6)
        collapse_term = max((qe_q1 - qe_q2) / max(qe_q1 + 1e-6, 1e-6), 0.0)
        req = {
            "id": row["id"],
            "row_id": row["id"],
            "dataset": row["dataset"],
            "source": row["source"],
            "metadata": row["metadata"],
            "request_id": request_id,
            "run_id": run_id,
            "subset_idx": 0,
            "student": row["mt_q1"],
            "selection": {
                "score_s": float(row["score_s"]),
                "qe_q1": qe_q1,
                "qe_q2": qe_q2,
                "delta_qe": float(delta_qe),
                "collapse_term": float(round(collapse_term, 6)),
            },
            "prompt_version": prompt_version,
            "prompt_hash": prompt_hash,
            "provider": provider,
            "model": model,
            "status": "ok",
            "config_hash": cfg_hash,
        }
        resp = {
            "id": row["id"],
            "row_id": row["id"],
            "dataset": row["dataset"],
            "source": row["source"],
            "metadata": row["metadata"],
            "request_id": request_id,
            "run_id": run_id,
            "subset_idx": 0,
            "provider": provider,
            "model": model,
            "status": "ok",
            "teacher_label": "minor_edit",
            "student": row["mt_q1"],
            "gold": f"KO_GOLD::{row['id']}",
            "reason": None,
            "prompt_version": prompt_version,
            "prompt_hash": prompt_hash,
            "usage": {
                "input_tokens": 96,
                "output_tokens": 72,
                "total_tokens": 168,
            },
            "cost": {
                "currency": "USD",
                "estimated": 0.0,
            },
            "latency_ms": 1.0,
            "attempt": 1,
            "error": None,
            "config_hash": cfg_hash,
        }
        requests.append(req)
        responses.append(resp)

    return requests, responses


def _derive_error_type_from_api_row(row: dict[str, Any]) -> str:
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


def _build_preference_pairs(api_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _assert_row_id_contract(
    input_rows: list[dict[str, Any]],
    q1_rows: list[dict[str, Any]],
    q2_rows: list[dict[str, Any]],
    scored_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    api_requests: list[dict[str, Any]],
    api_rows: list[dict[str, Any]],
    preference_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
) -> None:
    input_ids = [row["id"] for row in input_rows]
    for label, rows in (
        ("q1", q1_rows),
        ("q2", q2_rows),
        ("scored", scored_rows),
    ):
        ids = [row["id"] for row in rows]
        if ids != input_ids:
            raise SmokeValidationError(f"row_id drift detected in {label}.jsonl")

    selected_ids = [row["id"] for row in selected_rows]
    if any(row_id not in set(input_ids) for row_id in selected_ids):
        raise SmokeValidationError("selected.jsonl contains unknown row_id")

    request_ids = [row["id"] for row in api_requests]
    if request_ids != selected_ids:
        raise SmokeValidationError("api_requests.jsonl row_id mismatch")

    api_ids = [row["id"] for row in api_rows]
    if api_ids != selected_ids:
        raise SmokeValidationError("api.jsonl row_id mismatch")

    preference_ids = [row["id"] for row in preference_rows]
    if preference_ids != selected_ids:
        raise SmokeValidationError("preference_pairs.jsonl row_id mismatch")

    train_ids = [row["id"] for row in train_rows]
    if train_ids != selected_ids:
        raise SmokeValidationError("train_final rows must match selected rows")


def run_smoke(
    config_path: str = "configs/scp_stage4.yaml",
    overrides: list[str] | None = None,
    run_id_override: str | None = None,
    subset_size_override: int | None = None,
    use_prepared_data: bool = False,
) -> dict[str, Any]:
    cfg = compose_config(config_path, overrides=overrides)
    validate_config(cfg)

    run_id = run_id_override or str(_get_by_dotpath(cfg, "run.run_id", "local_contract"))
    root_dir = Path(str(_get_by_dotpath(cfg, "logging.local.root_dir", "artifacts/runs")))
    run_root = root_dir / run_id
    subset_root = run_root / "subsets" / "subset_000"
    subset_root.mkdir(parents=True, exist_ok=True)

    cfg_hash = compute_config_hash(cfg)
    persisted = persist_effective_config_artifacts(
        run_dir=run_root,
        effective_config=cfg,
        write_effective_config=bool(
            _get_by_dotpath(cfg, "logging.local.write_effective_config", True)
        ),
        write_config_hash=bool(
            _get_by_dotpath(cfg, "logging.local.write_config_hash", True)
        ),
    )
    if str(persisted["config_hash"]) != cfg_hash:
        raise SmokeValidationError("config_hash mismatch between stable hash and artifact hash")

    score_direction = str(
        _get_by_dotpath(cfg, "qe.primary.score_direction", "higher_is_better")
    )
    if score_direction not in {"higher_is_better", "lower_is_better"}:
        raise SmokeValidationError(
            "qe.primary.score_direction must be 'higher_is_better' or 'lower_is_better'"
        )

    pool_rows = _load_fixture_rows(use_prepared_data=use_prepared_data)
    input_rows = _select_subset(pool_rows, cfg, subset_size_override)
    q1_rows, q2_rows = _build_q_rows(input_rows, score_direction)
    scored_rows = _score_rows(q2_rows, score_direction)
    selected_rows = _select_fragile(scored_rows, cfg)
    api_requests, api_rows = _make_api_artifacts(selected_rows, run_id, cfg, cfg_hash)
    api_requests = validate_artifact_rows(api_requests, "api_requests")
    api_rows = validate_artifact_rows(api_rows, "api")
    preference_rows = _build_preference_pairs(api_rows)
    preference_rows = validate_artifact_rows(preference_rows, "preference_pairs")
    train_rows = [
        {
            "id": row["id"],
            "dataset": row["dataset"],
            "source": next(r["source"] for r in input_rows if r["id"] == row["id"]),
            "gold": row["gold"],
            "metadata": next(r["metadata"] for r in input_rows if r["id"] == row["id"]),
        }
        for row in api_rows
        if row["status"] == "ok"
    ]

    _assert_row_id_contract(
        input_rows,
        q1_rows,
        q2_rows,
        scored_rows,
        selected_rows,
        api_requests,
        api_rows,
        preference_rows,
        train_rows,
    )

    write_jsonl(subset_root / "input.jsonl", input_rows)
    write_jsonl(subset_root / "q1.jsonl", q1_rows)
    collapse_dir = subset_root / "collapse_adapter"
    collapse_dir.mkdir(parents=True, exist_ok=True)
    (collapse_dir / "collapse_state.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "adapter_path": str(collapse_dir),
                "run_id": run_id,
                "subset_idx": 0,
                "trained_rows": len(q1_rows),
                "runtime_mode": "mock",
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_jsonl(subset_root / "q2.jsonl", q2_rows)
    write_jsonl(subset_root / "scored.jsonl", scored_rows)
    write_jsonl(subset_root / "selected.jsonl", selected_rows)
    (subset_root / "clean_base.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "collapse_adapter": str(collapse_dir),
                "run_id": run_id,
                "subset_idx": 0,
                "verified_rows": len(q2_rows),
                "runtime_mode": "mock",
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_jsonl(subset_root / "api_requests.jsonl", api_requests)
    write_jsonl(subset_root / "api.jsonl", api_rows)
    write_jsonl(subset_root / "preference_pairs.jsonl", preference_rows)
    write_jsonl(run_root / "preference_pairs.jsonl", preference_rows)
    write_jsonl(subset_root / "train_final" / "train_rows.jsonl", train_rows)

    local_cfg = _get_by_dotpath(cfg, "logging.local", {})
    events_name = str(local_cfg.get("events_jsonl", "events.jsonl"))
    metrics_name = str(local_cfg.get("metrics_jsonl", "metrics.jsonl"))
    failures_name = str(local_cfg.get("failures_jsonl", "failures.jsonl"))
    logger = LocalJsonlLogger(
        run_root,
        events_name=events_name,
        metrics_name=metrics_name,
        failures_name=failures_name,
    )

    phase_artifacts = (
        ("infer-q1", "subsets/subset_000/q1.jsonl"),
        ("train-collapse-lora", "subsets/subset_000/collapse_adapter/collapse_state.json"),
        ("infer-q2", "subsets/subset_000/q2.jsonl"),
        ("score", "subsets/subset_000/scored.jsonl"),
        ("unload-collapse-lora", "subsets/subset_000/clean_base.json"),
        ("call-api", "subsets/subset_000/api.jsonl"),
        ("update-base", "subsets/subset_000/train_final/train_rows.jsonl"),
    )
    for phase, artifact_path in phase_artifacts:
        context = RequiredLogContext(
            run_id=run_id,
            subset_idx=0,
            phase=phase,
            config_hash=cfg_hash,
        )
        logger.log_event(
            context=context,
            event_type="phase_completed",
            status="ok",
            artifact_path=artifact_path,
        )

    summary = {
        "run_id": run_id,
        "config_hash": cfg_hash,
        "run_root": str(run_root),
        "counts": {
            "input": len(input_rows),
            "q1": len(q1_rows),
            "collapse_train": len(q1_rows),
            "q2": len(q2_rows),
            "scored": len(scored_rows),
            "selected": len(selected_rows),
            "clean_base": 1,
            "api_requests": len(api_requests),
            "api": len(api_rows),
            "preference_pairs": len(preference_rows),
            "train": len(train_rows),
        },
    }
    logger.log_metrics(
        context=RequiredLogContext(
            run_id=run_id,
            subset_idx=0,
            phase="smoke-local",
            config_hash=cfg_hash,
        ),
        metrics={
            "subset/input_rows": len(input_rows),
            "subset/q1_rows": len(q1_rows),
            "subset/q2_rows": len(q2_rows),
            "subset/scored_rows": len(scored_rows),
            "subset/selected_rows": len(selected_rows),
            "subset/api_ok_rows": len(api_rows),
            "subset/train_rows": len(train_rows),
        },
        metric_group="subset",
    )
    # Ensure failures layout exists even in all-success smoke runs.
    (run_root / failures_name).touch(exist_ok=True)
    (subset_root / failures_name).touch(exist_ok=True)

    (run_root / "smoke_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run local smoke subset flow")
    parser.add_argument("--config", default="configs/scp_stage4.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--use-prepared-data", action="store_true")
    args, overrides = parser.parse_known_args(argv)

    summary = run_smoke(
        config_path=args.config,
        overrides=overrides,
        run_id_override=args.run_id,
        subset_size_override=args.subset_size,
        use_prepared_data=args.use_prepared_data,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
