"""Local smoke pipeline for lightweight contract validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from scp_stage4.config.loader import compose_config, save_effective_config
from scp_stage4.config.validator import validate_config
from scp_stage4.pipeline.io_utils import iter_jsonl, write_jsonl


class SmokeValidationError(RuntimeError):
    """Raised when local smoke contract checks fail."""


def _get_by_dotpath(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    cursor: Any = cfg
    for part in key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, sub in value.items():
            lowered = key.lower()
            if "api_key" in lowered or "token" in lowered or "secret" in lowered:
                out[key] = "REDACTED"
            else:
                out[key] = _redact_secrets(sub)
        return out
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value


def _config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(_redact_secrets(cfg), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load_fixture_rows() -> list[dict[str, Any]]:
    candidates = [
        Path("tests/fixtures/datapool.train.jsonl"),
        Path("tests/fixtures/input.jsonl"),
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


def _build_q_rows(input_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    q1_rows: list[dict[str, Any]] = []
    q2_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(input_rows):
        qe_q1 = round(0.90 - (idx % 5) * 0.07, 6)
        collapse_drop = round(0.03 + (idx % 4) * 0.04, 6)
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


def _score_rows(q2_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in q2_rows:
        score_s = round(float(row["qe_q1"]) - float(row["qe_q2"]), 6)
        scored_row = dict(row)
        scored_row["score_s"] = score_s
        scored.append(scored_row)
    return scored


def _select_fragile(scored_rows: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    threshold = float(
        _get_by_dotpath(cfg, "qe.scoring.selection.default_rule.require_score_s_gte", 0.0)
    )
    top_fraction = float(
        _get_by_dotpath(cfg, "qe.scoring.selection.default_rule.top_fraction", 0.1)
    )

    eligible = [row for row in scored_rows if float(row["score_s"]) >= threshold]
    eligible_sorted = sorted(eligible, key=lambda r: (float(r["score_s"]), r["id"]), reverse=True)

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
    selected_rows: list[dict[str, Any]], run_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    requests: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []

    for row in selected_rows:
        request_id = f"{run_id}/subsets/subset_000/{row['id']}/api"
        req = {
            "id": row["id"],
            "dataset": row["dataset"],
            "source": row["source"],
            "metadata": row["metadata"],
            "request_id": request_id,
            "student": row["mt_q1"],
            "status": "ok",
        }
        resp = {
            "id": row["id"],
            "dataset": row["dataset"],
            "source": row["source"],
            "metadata": row["metadata"],
            "request_id": request_id,
            "status": "ok",
            "gold": f"KO_GOLD::{row['id']}",
        }
        requests.append(req)
        responses.append(resp)

    return requests, responses


def _assert_row_id_contract(
    input_rows: list[dict[str, Any]],
    q1_rows: list[dict[str, Any]],
    q2_rows: list[dict[str, Any]],
    scored_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    api_requests: list[dict[str, Any]],
    api_rows: list[dict[str, Any]],
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

    train_ids = [row["id"] for row in train_rows]
    if train_ids != selected_ids:
        raise SmokeValidationError("train_final rows must match selected rows")


def run_smoke(
    config_path: str = "configs/scp_stage4.yaml",
    overrides: list[str] | None = None,
    run_id_override: str | None = None,
    subset_size_override: int | None = None,
) -> dict[str, Any]:
    cfg = compose_config(config_path, overrides=overrides)
    validate_config(cfg)

    run_id = run_id_override or str(_get_by_dotpath(cfg, "run.run_id", "local_contract"))
    root_dir = Path(str(_get_by_dotpath(cfg, "logging.local.root_dir", "artifacts/runs")))
    run_root = root_dir / run_id
    subset_root = run_root / "subsets" / "subset_000"
    subset_root.mkdir(parents=True, exist_ok=True)

    cfg_hash = _config_hash(cfg)
    if bool(_get_by_dotpath(cfg, "logging.local.write_effective_config", True)):
        save_effective_config(cfg, run_root / "effective_config.yaml")
    if bool(_get_by_dotpath(cfg, "logging.local.write_config_hash", True)):
        (run_root / "config_hash.txt").write_text(cfg_hash + "\n", encoding="utf-8")

    pool_rows = _load_fixture_rows()
    input_rows = _select_subset(pool_rows, cfg, subset_size_override)
    q1_rows, q2_rows = _build_q_rows(input_rows)
    scored_rows = _score_rows(q2_rows)
    selected_rows = _select_fragile(scored_rows, cfg)
    api_requests, api_rows = _make_api_artifacts(selected_rows, run_id)
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
        train_rows,
    )

    write_jsonl(subset_root / "input.jsonl", input_rows)
    write_jsonl(subset_root / "q1.jsonl", q1_rows)
    write_jsonl(subset_root / "q2.jsonl", q2_rows)
    write_jsonl(subset_root / "scored.jsonl", scored_rows)
    write_jsonl(subset_root / "selected.jsonl", selected_rows)
    write_jsonl(subset_root / "api_requests.jsonl", api_requests)
    write_jsonl(subset_root / "api.jsonl", api_rows)
    write_jsonl(subset_root / "train_final" / "train_rows.jsonl", train_rows)

    events = [
        {
            "run_id": run_id,
            "subset_idx": 0,
            "phase": phase,
            "config_hash": cfg_hash,
            "event_type": "phase_completed",
            "status": "ok",
        }
        for phase in (
            "infer-q1",
            "train-collapse-lora",
            "infer-q2",
            "score",
            "call-api",
            "update-base",
        )
    ]
    write_jsonl(run_root / "events.jsonl", events)

    summary = {
        "run_id": run_id,
        "config_hash": cfg_hash,
        "run_root": str(run_root),
        "counts": {
            "input": len(input_rows),
            "q1": len(q1_rows),
            "q2": len(q2_rows),
            "scored": len(scored_rows),
            "selected": len(selected_rows),
            "api_requests": len(api_requests),
            "api": len(api_rows),
            "train": len(train_rows),
        },
    }
    (run_root / "smoke_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run local smoke subset flow")
    parser.add_argument("--config", default="configs/scp_stage4.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--subset-size", type=int, default=None)
    args, overrides = parser.parse_known_args(argv)

    summary = run_smoke(
        config_path=args.config,
        overrides=overrides,
        run_id_override=args.run_id,
        subset_size_override=args.subset_size,
    )
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
