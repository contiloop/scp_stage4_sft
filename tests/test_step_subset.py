from __future__ import annotations

import shutil
from pathlib import Path

from scp_stage4.data import read_jsonl
from scp_stage4.pipeline.prepare_data import run_prepare_data
from scp_stage4.pipeline.step_subset import (
    run_call_api,
    run_infer_q1,
    run_infer_q2,
    run_score,
    run_subset,
    run_update_base,
)


def _run_root(run_id: str) -> Path:
    return Path("artifacts/runs") / run_id


def _subset_root(run_id: str) -> Path:
    return _run_root(run_id) / "subsets" / "subset_000"


def _cleanup(run_id: str) -> None:
    root = _run_root(run_id)
    if root.exists():
        shutil.rmtree(root)


def test_run_subset_writes_stepwise_artifact_chain() -> None:
    run_id = "test_step_subset_run"
    _cleanup(run_id)
    try:
        run_prepare_data(config_path="configs/scp_stage4.yaml")
        summary = run_subset(
            config_path="configs/scp_stage4.yaml",
            run_id_override=run_id,
            subset_idx=0,
            subset_size_override=16,
            use_prepared_data=True,
        )

        subset_root = _subset_root(run_id)
        required = [
            _run_root(run_id) / "effective_config.yaml",
            _run_root(run_id) / "config_hash.txt",
            _run_root(run_id) / "events.jsonl",
            _run_root(run_id) / "metrics.jsonl",
            _run_root(run_id) / "failures.jsonl",
            _run_root(run_id) / "run_subset_summary.json",
            subset_root / "input.jsonl",
            subset_root / "q1.jsonl",
            subset_root / "q2.jsonl",
            subset_root / "scored.jsonl",
            subset_root / "selected.jsonl",
            subset_root / "api_requests.jsonl",
            subset_root / "api.jsonl",
            subset_root / "events.jsonl",
            subset_root / "metrics.jsonl",
            subset_root / "failures.jsonl",
            subset_root / "train_final" / "train_rows.jsonl",
        ]
        for path in required:
            assert path.exists(), f"missing artifact: {path}"

        input_rows = read_jsonl(subset_root / "input.jsonl")
        q1_rows = read_jsonl(subset_root / "q1.jsonl")
        q2_rows = read_jsonl(subset_root / "q2.jsonl")
        scored_rows = read_jsonl(subset_root / "scored.jsonl")
        selected_rows = read_jsonl(subset_root / "selected.jsonl")
        api_requests = read_jsonl(subset_root / "api_requests.jsonl")
        api_rows = read_jsonl(subset_root / "api.jsonl")
        train_rows = read_jsonl(subset_root / "train_final" / "train_rows.jsonl")

        input_ids = [row["id"] for row in input_rows]
        assert [row["id"] for row in q1_rows] == input_ids
        assert [row["id"] for row in q2_rows] == input_ids
        assert [row["id"] for row in scored_rows] == input_ids

        selected_ids = [row["id"] for row in selected_rows]
        assert set(selected_ids).issubset(set(input_ids))
        assert [row["id"] for row in api_requests] == selected_ids
        assert [row["id"] for row in api_rows] == selected_ids
        assert [row["id"] for row in train_rows] == selected_ids

        assert summary["counts"]["q1"] == len(q1_rows)
        assert summary["counts"]["q2"] == len(q2_rows)
        assert summary["counts"]["selected"] == len(selected_rows)
    finally:
        _cleanup(run_id)


def test_step_entrypoints_run_in_sequence_and_update_base_filters_non_ok() -> None:
    run_id = "test_step_subset_sequence"
    _cleanup(run_id)
    try:
        run_prepare_data(config_path="configs/scp_stage4.yaml")
        run_infer_q1(
            config_path="configs/scp_stage4.yaml",
            run_id_override=run_id,
            subset_idx=0,
            subset_size_override=12,
            use_prepared_data=True,
        )
        run_infer_q2(
            config_path="configs/scp_stage4.yaml",
            run_id_override=run_id,
            subset_idx=0,
        )
        run_score(
            config_path="configs/scp_stage4.yaml",
            run_id_override=run_id,
            subset_idx=0,
        )
        run_call_api(
            config_path="configs/scp_stage4.yaml",
            run_id_override=run_id,
            subset_idx=0,
        )

        subset_root = _subset_root(run_id)
        api_path = subset_root / "api.jsonl"
        api_rows = read_jsonl(api_path)
        assert api_rows, "api rows should exist"

        api_rows[0]["status"] = "failed"
        api_rows[0]["gold"] = "KO_GOLD::FAILED_PLACEHOLDER"
        from scp_stage4.data import write_jsonl

        write_jsonl(api_path, api_rows)

        update_summary = run_update_base(
            config_path="configs/scp_stage4.yaml",
            run_id_override=run_id,
            subset_idx=0,
        )

        train_rows = read_jsonl(subset_root / "train_final" / "train_rows.jsonl")
        assert update_summary["train_rows"] == len(train_rows)
        assert len(train_rows) == len([row for row in api_rows if row["status"] == "ok"])
        assert all(row["id"] != api_rows[0]["id"] for row in train_rows)

        selected_rows = read_jsonl(subset_root / "selected.jsonl")
        ranks = [row["selection_rank"] for row in selected_rows]
        assert all(isinstance(rank, int) for rank in ranks)
        assert min(ranks) == 1
    finally:
        _cleanup(run_id)
