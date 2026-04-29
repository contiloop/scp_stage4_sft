from __future__ import annotations

import json
import shutil
import sys
import tarfile
from pathlib import Path

import pytest

from scp_stage4.data import read_jsonl
from scp_stage4.pipeline.prepare_data import run_prepare_data
from scp_stage4.pipeline.step_subset import (
    StepSubsetError,
    run_call_api,
    run_infer_q1,
    run_infer_q2,
    run_score,
    run_stage,
    run_subset,
    run_train_collapse_lora,
    run_unload_collapse_lora,
    run_update_base,
    main as step_subset_main,
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
            subset_root / "collapse_adapter" / "collapse_state.json",
            subset_root / "q2.jsonl",
            subset_root / "scored.jsonl",
            subset_root / "selected.jsonl",
            subset_root / "clean_base.json",
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
        run_train_collapse_lora(
            config_path="configs/scp_stage4.yaml",
            run_id_override=run_id,
            subset_idx=0,
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
        run_unload_collapse_lora(
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
        api_rows[0]["gold"] = None
        api_rows[0]["reason"] = "forced test failure row"
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


def test_run_subset_with_subprocess_runtimes() -> None:
    run_id = "test_step_subset_subprocess"
    _cleanup(run_id)
    try:
        run_prepare_data(config_path="configs/scp_stage4.yaml")
        inference_cmd = json.dumps(
            [sys.executable, "-m", "scp_stage4.pipeline.workers.mock_inference_worker"]
        )
        qe_cmd = json.dumps([sys.executable, "-m", "scp_stage4.pipeline.workers.mock_qe_worker"])
        api_cmd = json.dumps([sys.executable, "-m", "scp_stage4.pipeline.workers.mock_api_worker"])
        training_cmd = json.dumps(
            [sys.executable, "-m", "scp_stage4.pipeline.workers.mock_training_worker"]
        )

        summary = run_subset(
            config_path="configs/scp_stage4.yaml",
            run_id_override=run_id,
            subset_idx=0,
            subset_size_override=8,
            use_prepared_data=True,
            overrides=[
                "inference.runtime.mode=subprocess",
                f"inference.runtime.subprocess.command={inference_cmd}",
                "qe.runtime.mode=subprocess",
                f"qe.runtime.subprocess.command={qe_cmd}",
                "external_api.runtime.mode=subprocess",
                f"external_api.runtime.subprocess.command={api_cmd}",
                "training.runtime.mode=subprocess",
                f"training.runtime.subprocess.collapse_command={training_cmd}",
                f"training.runtime.subprocess.unload_command={training_cmd}",
                f"training.runtime.subprocess.update_command={training_cmd}",
            ],
        )

        subset_root = _subset_root(run_id)
        q1_rows = read_jsonl(subset_root / "q1.jsonl")
        q2_rows = read_jsonl(subset_root / "q2.jsonl")
        api_rows = read_jsonl(subset_root / "api.jsonl")
        runtime_io = subset_root / "runtime_io"

        assert q1_rows and q2_rows and api_rows
        assert (runtime_io / "infer-q1.input.jsonl").exists()
        assert (runtime_io / "infer-q1.output.jsonl").exists()
        assert (runtime_io / "qe-q1.input.jsonl").exists()
        assert (runtime_io / "qe-q2.output.jsonl").exists()
        assert (runtime_io / "train-collapse-lora.output.jsonl").exists()
        assert (runtime_io / "unload-collapse-lora.output.jsonl").exists()
        assert (runtime_io / "call-api.output.jsonl").exists()
        assert (runtime_io / "update-base.output.jsonl").exists()
        assert summary["counts"]["api"] == len(api_rows)

        assert all(str(row["mt_q1"]).startswith("KO_Q1::") for row in q1_rows)
        assert all(str(row["mt_q2"]).startswith("KO_Q2::") for row in q2_rows)
        assert all(str(row["gold"]).startswith("KO_GOLD::") for row in api_rows)
    finally:
        _cleanup(run_id)


def test_infer_q2_requires_collapse_adapter_state() -> None:
    run_id = "test_step_subset_require_collapse_before_q2"
    _cleanup(run_id)
    try:
        run_prepare_data(config_path="configs/scp_stage4.yaml")
        run_infer_q1(
            config_path="configs/scp_stage4.yaml",
            run_id_override=run_id,
            subset_idx=0,
            subset_size_override=4,
            use_prepared_data=True,
        )
        try:
            run_infer_q2(
                config_path="configs/scp_stage4.yaml",
                run_id_override=run_id,
                subset_idx=0,
            )
            assert False, "infer-q2 must fail when collapse adapter state is missing"
        except StepSubsetError as exc:
            assert "collapse adapter state is missing" in str(exc)
    finally:
        _cleanup(run_id)


def test_step_subset_cli_writes_structured_failure_log_on_error() -> None:
    run_id = "test_step_subset_cli_failure_logging"
    _cleanup(run_id)
    try:
        rc = step_subset_main(
            [
                "call-api",
                "--config",
                "configs/scp_stage4.yaml",
                "--run-id",
                run_id,
                "--subset-idx",
                "0",
            ]
        )
        assert rc == 1

        run_root = _run_root(run_id)
        failures_path = run_root / "failures.jsonl"
        subset_failures_path = _subset_root(run_id) / "failures.jsonl"
        assert failures_path.exists()
        assert subset_failures_path.exists()

        failure_rows = read_jsonl(failures_path)
        assert failure_rows, "expected at least one structured failure row"
        latest = failure_rows[-1]
        assert latest["run_id"] == run_id
        assert latest["subset_idx"] == 0
        assert latest["phase"] == "call-api"
        assert latest["status"] == "failed"
        assert latest["failure_type"] == "call-api_failed"
        assert isinstance(latest["config_hash"], str) and latest["config_hash"]
    finally:
        _cleanup(run_id)


def test_run_subset_use_prepared_data_requires_prepare_data(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(
        StepSubsetError,
        match="No prepared train rows found; run prepare-data before using prepared-data mode",
    ):
        run_subset(
            config_path=str(Path(__file__).resolve().parents[1] / "configs" / "scp_stage4.yaml"),
            run_id_override="test_missing_prepared_rows",
            subset_idx=0,
            subset_size_override=8,
            use_prepared_data=True,
        )


def test_run_subset_writes_subset_archive_when_enabled() -> None:
    run_id = "test_step_subset_archive_enabled"
    _cleanup(run_id)
    try:
        run_prepare_data(config_path="configs/scp_stage4.yaml")
        summary = run_subset(
            config_path="configs/scp_stage4.yaml",
            run_id_override=run_id,
            subset_idx=0,
            subset_size_override=8,
            use_prepared_data=True,
            overrides=["pipeline.stage.subset_archive.enabled=true"],
        )
        archive = summary.get("subset_archive")
        assert isinstance(archive, dict)
        archive_path = Path(str(archive["archive_path"]))
        manifest_path = Path(str(archive["manifest_path"]))
        assert archive_path.exists()
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["run_id"] == run_id
        assert manifest["subset_idx"] == 0
        assert manifest["file_count"] >= 1
        with tarfile.open(archive_path, "r:gz") as handle:
            names = handle.getnames()
            assert any(name.endswith("subset_000/q1.jsonl") for name in names)
            assert any(name.endswith("subset_000/train_final/train_rows.jsonl") for name in names)
    finally:
        _cleanup(run_id)


def test_run_stage_can_prune_subset_dirs_after_archiving() -> None:
    run_id = "test_stage_archive_prune"
    _cleanup(run_id)
    try:
        run_prepare_data(config_path="configs/scp_stage4.yaml")
        summary = run_stage(
            config_path="configs/scp_stage4.yaml",
            run_id_override=run_id,
            subset_size_override=8,
            overrides=[
                "pipeline.stage.max_subsets=1",
                "pipeline.stage.subset_archive.enabled=true",
                "pipeline.stage.subset_archive.delete_original_after_archive=true",
            ],
        )
        assert summary["archived_subset_dirs_pruned"] == 1
        subset_root = _subset_root(run_id)
        assert (subset_root / "ARCHIVED.json").exists()
        assert not (subset_root / "q1.jsonl").exists()

        archive_path = _run_root(run_id) / "archives" / "subsets" / "subset_000.tar.gz"
        manifest_path = _run_root(run_id) / "archives" / "subsets" / "subset_000.manifest.json"
        assert archive_path.exists()
        assert manifest_path.exists()
    finally:
        _cleanup(run_id)
