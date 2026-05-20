from __future__ import annotations

import json
import shutil
import sys
import tarfile
from pathlib import Path

import pytest

import scp_stage4.pipeline.step_subset as step_subset_mod
from scp_stage4.data import read_jsonl, write_jsonl
from scp_stage4.pipeline.prepare_data import run_prepare_data
from scp_stage4.pipeline.step_subset import (
    StepSubsetError,
    run_call_api,
    run_infer_q1,
    run_infer_q2,
    run_score,
    run_eval_ood,
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


def test_run_score_default_selects_absolute_relative_delta_qe() -> None:
    run_id = "test_score_abs_relative_delta"
    _cleanup(run_id)
    try:
        subset_root = _subset_root(run_id)
        subset_root.mkdir(parents=True, exist_ok=True)
        metadata = {
            "title": None,
            "document_type": "article",
            "text_role": "body",
            "original_id": None,
            "parent_id": None,
            "chunk_idx": None,
        }
        write_jsonl(
            subset_root / "q2.jsonl",
            [
                {
                    "id": "improved",
                    "dataset": "fixture",
                    "source": "source improved",
                    "metadata": metadata,
                    "mt_q1": "q1",
                    "mt_q2": "q2",
                    "qe_q1": 0.80,
                    "qe_raw_q1": 0.80,
                    "metricx_q1_clamped": False,
                    "qe_q2": 0.92,
                    "qe_raw_q2": 0.92,
                    "metricx_q2_clamped": False,
                },
                {
                    "id": "degraded",
                    "dataset": "fixture",
                    "source": "source degraded",
                    "metadata": metadata,
                    "mt_q1": "q1",
                    "mt_q2": "q2",
                    "qe_q1": 0.80,
                    "qe_raw_q1": 0.80,
                    "metricx_q1_clamped": False,
                    "qe_q2": 0.74,
                    "qe_raw_q2": 0.74,
                    "metricx_q2_clamped": False,
                },
                {
                    "id": "stable",
                    "dataset": "fixture",
                    "source": "source stable",
                    "metadata": metadata,
                    "mt_q1": "q1",
                    "mt_q2": "q2",
                    "qe_q1": 0.80,
                    "qe_raw_q1": 0.80,
                    "metricx_q1_clamped": False,
                    "qe_q2": 0.81,
                    "qe_raw_q2": 0.81,
                    "metricx_q2_clamped": False,
                },
            ],
        )

        run_score(
            config_path="configs/scp_stage4.yaml",
            overrides=[
                "qe.scoring.weighted_score.alpha=0.0",
                "qe.scoring.weighted_score.beta=1.0",
                "qe.scoring.selection.default_rule.top_fraction=0.33",
            ],
            run_id_override=run_id,
            subset_idx=0,
        )

        scored = {row["id"]: row for row in read_jsonl(subset_root / "scored.jsonl")}
        selected = read_jsonl(subset_root / "selected.jsonl")
        assert scored["improved"]["delta_qe"] == 0.12
        assert scored["improved"]["collapse_term"] == 0.15
        assert scored["improved"]["collapse_term_type"] == "abs_relative_delta"
        assert scored["degraded"]["collapse_term"] == 0.075
        assert [row["id"] for row in selected] == ["improved"]
        assert selected[0]["collapse_term_type"] == "abs_relative_delta"
    finally:
        _cleanup(run_id)


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
            _run_root(run_id) / "preference_pairs.jsonl",
            _run_root(run_id) / "ood_eval.jsonl",
            _run_root(run_id) / "run_subset_summary.json",
            _run_root(run_id) / "eval" / "ood_test" / "subset_000.rows.jsonl",
            _run_root(run_id) / "eval" / "ood_test" / "subset_000.summary.json",
            subset_root / "input.jsonl",
            subset_root / "q1.jsonl",
            subset_root / "collapse_adapter" / "collapse_state.json",
            subset_root / "q2.jsonl",
            subset_root / "scored.jsonl",
            subset_root / "selected.jsonl",
            subset_root / "clean_base.json",
            subset_root / "api_requests.jsonl",
            subset_root / "api.jsonl",
            subset_root / "preference_pairs.jsonl",
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
        preference_rows = read_jsonl(subset_root / "preference_pairs.jsonl")
        run_preference_rows = read_jsonl(_run_root(run_id) / "preference_pairs.jsonl")
        train_rows = read_jsonl(subset_root / "train_final" / "train_rows.jsonl")

        input_ids = [row["id"] for row in input_rows]
        assert [row["id"] for row in q1_rows] == input_ids
        assert [row["id"] for row in q2_rows] == input_ids
        assert [row["id"] for row in scored_rows] == input_ids

        selected_ids = [row["id"] for row in selected_rows]
        assert set(selected_ids).issubset(set(input_ids))
        assert [row["id"] for row in api_requests] == selected_ids
        assert [row["id"] for row in api_rows] == selected_ids
        assert [row["id"] for row in preference_rows] == selected_ids
        assert [row["id"] for row in run_preference_rows] == selected_ids
        assert [row["id"] for row in train_rows] == selected_ids

        assert summary["counts"]["q1"] == len(q1_rows)
        assert summary["counts"]["q2"] == len(q2_rows)
        assert summary["counts"]["selected"] == len(selected_rows)
        assert summary["preference_pairs_run_total"] == len(run_preference_rows)
        assert "ood_eval" in summary
    finally:
        _cleanup(run_id)


def test_run_eval_ood_writes_metricx_bleu_chrf_artifacts() -> None:
    run_id = "test_step_subset_eval_ood"
    _cleanup(run_id)
    try:
        run_prepare_data(config_path="configs/scp_stage4.yaml")
        run_subset(
            config_path="configs/scp_stage4.yaml",
            run_id_override=run_id,
            subset_idx=0,
            subset_size_override=8,
            use_prepared_data=True,
        )
        summary = run_eval_ood(
            config_path="configs/scp_stage4.yaml",
            run_id_override=run_id,
            subset_idx=0,
        )
        run_root = _run_root(run_id)
        rows_path = run_root / "eval" / "ood_test" / "subset_000.rows.jsonl"
        summary_path = run_root / "eval" / "ood_test" / "subset_000.summary.json"
        history_path = run_root / "eval" / "ood_test" / "history.jsonl"
        monitor_path = run_root / "ood_eval.jsonl"
        best_path = run_root / "checkpoints" / "best.json"
        assert rows_path.exists()
        assert summary_path.exists()
        assert history_path.exists()
        assert monitor_path.exists()
        assert best_path.exists()

        rows = read_jsonl(rows_path)
        assert rows, "ood eval rows should not be empty"
        first = rows[0]
        assert "mt" in first
        assert "xcomet" in first
        assert "bleu" in first
        assert "chrf" in first
        assert summary["rows"] == len(rows)

        eval_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert "xcomet_mean" in eval_summary
        assert "bleu_mean" in eval_summary
        assert "chrf_mean" in eval_summary

        monitor_rows = read_jsonl(monitor_path)
        assert len(monitor_rows) == 1
        monitor = monitor_rows[0]
        assert monitor["run_id"] == run_id
        assert monitor["subset_idx"] == 0
        assert set(monitor) == {"run_id", "subset_idx", "metrics"}
        assert monitor["metrics"]["ood/rows"] == len(rows)
        assert "ood/xcomet_mean" in monitor["metrics"]

        best = json.loads(best_path.read_text(encoding="utf-8"))
        assert best["status"] == "ok"
        assert best["run_id"] == run_id
        assert best["subset_idx"] == 0
        assert best["metric_key"] == "ood/xcomet_mean"
        assert best["metric_value"] == monitor["metrics"]["ood/xcomet_mean"]
        assert best["checkpoint_path"] == str(
            run_root / "subsets" / "subset_000" / "train_final" / "main_adapter"
        )

        run_eval_ood(
            config_path="configs/scp_stage4.yaml",
            run_id_override=run_id,
            subset_idx=0,
        )
        assert len(read_jsonl(monitor_path)) == 1
    finally:
        _cleanup(run_id)


def test_run_eval_ood_uses_eval_decoding_not_q1_sampling() -> None:
    run_id = "test_step_subset_eval_ood_greedy_decoding"
    _cleanup(run_id)
    try:
        run_prepare_data(config_path="configs/scp_stage4.yaml")
        inference_cmd = json.dumps(
            [sys.executable, "-m", "scp_stage4.pipeline.workers.mock_inference_worker"]
        )
        run_eval_ood(
            config_path="configs/scp_stage4.yaml",
            run_id_override=run_id,
            subset_idx=0,
            overrides=[
                "inference.runtime.mode=subprocess",
                f"inference.runtime.subprocess.command={inference_cmd}",
                "inference.q1.do_sample=true",
                "inference.q1.temperature=1.1",
                "inference.eval.do_sample=false",
                "inference.eval.temperature=0.0",
                "inference.eval.top_p=null",
            ],
        )
        request_rows = read_jsonl(
            _subset_root(run_id) / "runtime_io" / "infer-ood.input.jsonl"
        )
        assert request_rows
        decoding = request_rows[0]["decoding"]
        assert decoding["do_sample"] is False
        assert decoding["temperature"] == 0.0
        assert decoding["top_p"] is None
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


def test_run_subset_with_multi_gpu_inference_shards_merges_deterministically() -> None:
    run_id = "test_step_subset_multi_gpu_infer_shards"
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

        run_subset(
            config_path="configs/scp_stage4.yaml",
            run_id_override=run_id,
            subset_idx=0,
            subset_size_override=11,
            use_prepared_data=False,
            overrides=[
                "inference.runtime.mode=subprocess",
                f"inference.runtime.subprocess.command={inference_cmd}",
                "inference.runtime.multi_gpu.enabled=true",
                "inference.runtime.multi_gpu.gpu_ids=[0,1]",
                "inference.runtime.multi_gpu.shard_strategy=row_id_hash",
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
        runtime_io = subset_root / "runtime_io"
        input_rows = read_jsonl(subset_root / "input.jsonl")
        q1_rows = read_jsonl(subset_root / "q1.jsonl")
        q2_rows = read_jsonl(subset_root / "q2.jsonl")

        assert len(input_rows) > 1
        assert [row["id"] for row in q1_rows] == [row["id"] for row in input_rows]
        assert [row["id"] for row in q2_rows] == [row["id"] for row in input_rows]
        assert all(str(row["mt_q1"]).startswith("KO_Q1::") for row in q1_rows)
        assert all(str(row["mt_q2"]).startswith("KO_Q2::") for row in q2_rows)

        q1_input_parts = sorted(runtime_io.glob("infer-q1.input.part*.jsonl"))
        q1_output_parts = sorted(runtime_io.glob("infer-q1.output.part*.jsonl"))
        q2_input_parts = sorted(runtime_io.glob("infer-q2.input.part*.jsonl"))
        q2_output_parts = sorted(runtime_io.glob("infer-q2.output.part*.jsonl"))

        assert q1_input_parts, "infer-q1 shard input parts should exist"
        assert q1_output_parts, "infer-q1 shard output parts should exist"
        assert q2_input_parts, "infer-q2 shard input parts should exist"
        assert q2_output_parts, "infer-q2 shard output parts should exist"
        assert len(q1_input_parts) <= 2
        assert len(q2_input_parts) <= 2
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


def test_run_subset_prefers_parquet_before_jsonl(tmp_path: Path, monkeypatch) -> None:
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")
    from scp_stage4.data import write_jsonl

    monkeypatch.chdir(tmp_path)
    config_path = str(Path(__file__).resolve().parents[1] / "configs" / "scp_stage4.yaml")
    run_prepare_data(
        config_path=config_path,
        overrides=[
            "data.split.eval_ratio=0",
            "data.subset_size=2",
        ],
    )

    train_parquet_path = Path("artifacts/data/datapool.train.parquet")
    train_jsonl_path = Path("artifacts/data/datapool.train.jsonl")
    parquet_rows = pyarrow_parquet.read_table(train_parquet_path).to_pylist()
    assert parquet_rows
    expected_first_id = str(parquet_rows[0]["id"])

    write_jsonl(
        train_jsonl_path,
        [
            {
                "id": "jsonl_priority_probe_row",
                "dataset": "jsonl_probe",
                "source": "JSONL probe row should not be used when parquet exists.",
                "metadata": {
                    "title": None,
                    "document_type": "other",
                    "text_role": "body",
                    "original_id": "probe-1",
                    "parent_id": None,
                    "chunk_idx": None,
                },
            }
        ],
        ensure_ascii=False,
    )

    run_id = "test_parquet_priority"
    summary = run_subset(
        config_path=config_path,
        run_id_override=run_id,
        subset_idx=0,
        subset_size_override=1,
        use_prepared_data=True,
        overrides=["pipeline.subset.shuffle=false"],
    )
    assert int(summary["counts"]["input"]) == 1

    input_rows = read_jsonl(_subset_root(run_id) / "input.jsonl")
    assert input_rows
    assert input_rows[0]["id"] == expected_first_id


def test_run_subset_limits_prepared_row_loading_when_shuffle_disabled() -> None:
    run_id = "test_step_subset_load_limit"
    _cleanup(run_id)
    try:
        run_prepare_data(config_path="configs/scp_stage4.yaml")
        captured_limits: list[int | None] = []
        original_loader = step_subset_mod._load_prepared_rows

        def _capturing_loader(path: Path, *, max_rows: int | None = None):
            captured_limits.append(max_rows)
            return original_loader(path, max_rows=max_rows)

        from unittest.mock import patch

        with patch(
            "scp_stage4.pipeline.step_subset._load_prepared_rows",
            _capturing_loader,
        ):
            run_subset(
                config_path="configs/scp_stage4.yaml",
                run_id_override=run_id,
                subset_idx=0,
                subset_size_override=4,
                use_prepared_data=True,
                overrides=["pipeline.subset.shuffle=false"],
            )

        assert captured_limits, "expected at least one prepared-row load attempt"
        assert captured_limits[0] == 4
    finally:
        _cleanup(run_id)


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


def test_run_stage_runs_eval_after_subset_when_enabled() -> None:
    run_id = "test_stage_eval_after_subset"
    _cleanup(run_id)
    try:
        run_prepare_data(config_path="configs/scp_stage4.yaml")
        summary = run_stage(
            config_path="configs/scp_stage4.yaml",
            run_id_override=run_id,
            subset_size_override=8,
            overrides=["pipeline.stage.max_subsets=1"],
        )
        assert summary["subsets_run"] == 1
        assert summary["ood_evals_run"] == 1
        assert "ood_eval_summaries" in summary and len(summary["ood_eval_summaries"]) == 1
        subset_summary = summary["subsets"][0]
        assert "ood_eval" in subset_summary
        assert (_run_root(run_id) / "eval" / "ood_test" / "subset_000.summary.json").exists()
    finally:
        _cleanup(run_id)


def test_checkpoint_retention_keeps_only_recent_two_subsets(tmp_path: Path, monkeypatch) -> None:
    run_id = "test_checkpoint_retention_last_two"
    _cleanup(run_id)
    try:
        monkeypatch.chdir(tmp_path)
        config_path = str(Path(__file__).resolve().parents[1] / "configs" / "scp_stage4.yaml")
        from scp_stage4.data import write_jsonl

        data_dir = Path("artifacts/data")
        data_dir.mkdir(parents=True, exist_ok=True)
        train_rows = []
        for idx in range(12):
            train_rows.append(
                {
                    "id": f"retention_{idx:03d}",
                    "dataset": "retention_test",
                    "source": f"source sentence {idx}",
                    "metadata": {
                        "title": None,
                        "document_type": "other",
                        "text_role": "body",
                        "original_id": f"orig_{idx:03d}",
                        "parent_id": None,
                        "chunk_idx": None,
                    },
                }
            )
        write_jsonl(data_dir / "datapool.train.jsonl", train_rows, ensure_ascii=False)

        for subset_idx in range(3):
            run_subset(
                config_path=config_path,
                run_id_override=run_id,
                subset_idx=subset_idx,
                subset_size_override=4,
                use_prepared_data=True,
                use_sampled_data=False,
                overrides=["pipeline.subset.shuffle=false"],
            )

        subset0_train = _run_root(run_id) / "subsets" / "subset_000" / "train_final"
        subset1_train = _run_root(run_id) / "subsets" / "subset_001" / "train_final"
        subset2_train = _run_root(run_id) / "subsets" / "subset_002" / "train_final"

        assert not (subset0_train / "main_adapter").exists()
        assert (subset0_train / "PRUNED_CHECKPOINTS.json").exists()
        assert (subset1_train / "main_adapter").exists()
        assert (subset2_train / "main_adapter").exists()
    finally:
        _cleanup(run_id)


def test_checkpoint_retention_keeps_one_best_plus_last_plus_current(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "test_checkpoint_retention_best_plus_last"
    _cleanup(run_id)
    try:
        monkeypatch.chdir(tmp_path)
        config_path = str(Path(__file__).resolve().parents[1] / "configs" / "scp_stage4.yaml")
        from scp_stage4.data import write_jsonl

        data_dir = Path("artifacts/data")
        data_dir.mkdir(parents=True, exist_ok=True)
        train_rows = []
        for idx in range(12):
            train_rows.append(
                {
                    "id": f"retention_best_{idx:03d}",
                    "dataset": "retention_test",
                    "source": f"source sentence {idx}",
                    "metadata": {
                        "title": None,
                        "document_type": "other",
                        "text_role": "body",
                        "original_id": f"orig_best_{idx:03d}",
                        "parent_id": None,
                        "chunk_idx": None,
                    },
                }
            )
        write_jsonl(data_dir / "datapool.train.jsonl", train_rows, ensure_ascii=False)

        def _append_eval_metric(subset_idx: int, quality: float) -> None:
            run_root = _run_root(run_id)
            run_root.mkdir(parents=True, exist_ok=True)
            record = {
                "run_id": run_id,
                "subset_idx": subset_idx,
                "phase": "eval-ood",
                "status": "ok",
                "metric_group": "ood_eval",
                "metrics": {"ood/xcomet_mean": quality},
            }
            with (run_root / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")

        run_subset(
            config_path=config_path,
            run_id_override=run_id,
            subset_idx=0,
            subset_size_override=4,
            use_prepared_data=True,
            use_sampled_data=False,
            run_eval_after_subset=False,
            overrides=[
                "pipeline.subset.shuffle=false",
                "training.checkpoint.keep_last_n=1",
                "training.checkpoint.keep_best_n=1",
            ],
        )
        _append_eval_metric(0, 23.0)

        run_subset(
            config_path=config_path,
            run_id_override=run_id,
            subset_idx=1,
            subset_size_override=4,
            use_prepared_data=True,
            use_sampled_data=False,
            run_eval_after_subset=False,
            overrides=[
                "pipeline.subset.shuffle=false",
                "training.checkpoint.keep_last_n=1",
                "training.checkpoint.keep_best_n=1",
            ],
        )
        _append_eval_metric(1, 10.0)

        run_subset(
            config_path=config_path,
            run_id_override=run_id,
            subset_idx=2,
            subset_size_override=4,
            use_prepared_data=True,
            use_sampled_data=False,
            run_eval_after_subset=False,
            overrides=[
                "pipeline.subset.shuffle=false",
                "training.checkpoint.keep_last_n=1",
                "training.checkpoint.keep_best_n=1",
            ],
        )

        subset0_train = _run_root(run_id) / "subsets" / "subset_000" / "train_final"
        subset1_train = _run_root(run_id) / "subsets" / "subset_001" / "train_final"
        subset2_train = _run_root(run_id) / "subsets" / "subset_002" / "train_final"

        assert (subset0_train / "main_adapter").exists()
        assert (subset1_train / "main_adapter").exists()
        assert (subset2_train / "main_adapter").exists()
        assert not (subset0_train / "PRUNED_CHECKPOINTS.json").exists()
    finally:
        _cleanup(run_id)
