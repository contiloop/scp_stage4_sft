from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scp_stage4.artifacts.config_artifacts import compute_config_hash, persist_effective_config_artifacts
from scp_stage4.logging.local import LocalJsonlLogger
from scp_stage4.logging import local as local_logging_module
from scp_stage4.logging.schema import RequiredLogContext, build_event_record, build_failure_record


def test_valid_event_and_failure_serialization() -> None:
    context = RequiredLogContext(
        run_id="run_abc123",
        subset_idx=0,
        phase="call-api",
        config_hash="abc123hash",
    )

    event = build_event_record(
        context=context,
        event_type="phase_completed",
        status="ok",
        metrics={"selected_rows": 8, "score_mean": 0.82},
        artifact_path="subsets/subset_000/scored.jsonl",
        error="api_key=sk-secret-value-123456",
    )
    assert event["status"] == "ok"
    assert event["run_id"] == "run_abc123"
    assert event["subset_idx"] == 0
    assert event["phase"] == "call-api"
    assert event["config_hash"] == "abc123hash"
    assert "[REDACTED]" in event["error"]
    assert "sk-secret-value-123456" not in event["error"]
    json.dumps(event)

    failure = build_failure_record(
        context=context,
        failure_type="api_timeout",
        status="failed",
        row_id="sample_000001",
        request_id="run_abc123/subsets/subset_000/sample_000001/api",
        provider="openai",
        model="configured-provider-model",
        attempt=3,
        error="Bearer verysensitiveverysensitive",
    )
    assert failure["status"] == "failed"
    assert failure["failure_type"] == "api_timeout"
    assert failure["row_id"] == "sample_000001"
    assert failure["attempt"] == 3
    assert "Bearer [REDACTED]" in failure["error"]
    json.dumps(failure)


def test_missing_required_fields_fail() -> None:
    with pytest.raises(ValueError):
        RequiredLogContext(run_id="", subset_idx=0, phase="score", config_hash="hash")

    with pytest.raises(ValueError):
        RequiredLogContext(run_id="run_abc", subset_idx=0, phase="", config_hash="hash")

    with pytest.raises(ValueError):
        RequiredLogContext(run_id="run_abc", subset_idx=0, phase="score", config_hash="")

    with pytest.raises(ValueError):
        RequiredLogContext(run_id="run_abc", subset_idx=-1, phase="score", config_hash="hash")

    context = RequiredLogContext(run_id="run_abc", subset_idx=0, phase="score", config_hash="hash")
    with pytest.raises(ValueError):
        build_event_record(context=context, event_type="", status="ok")

    with pytest.raises(ValueError):
        build_failure_record(context=context, failure_type="", status="failed")


def test_config_hash_deterministic_and_secret_redaction(tmp_path: Path) -> None:
    config_a = {
        "run": {"seed": 42},
        "data": {"subset_size": 32},
        "external_api": {
            "primary": {
                "provider": "openai",
                "api_key": "sk-live-a-1234567890",
                "api_key_env": "OPENAI_API_KEY",
            }
        },
    }
    config_b = {
        "external_api": {
            "primary": {
                "api_key_env": "OPENAI_API_KEY",
                "api_key": "sk-live-b-0987654321",
                "provider": "openai",
            }
        },
        "data": {"subset_size": 32},
        "run": {"seed": 42},
    }

    hash_a = compute_config_hash(config_a)
    hash_b = compute_config_hash(config_b)
    assert hash_a == hash_b

    config_c = {
        "run": {"seed": 42},
        "data": {"subset_size": 64},
        "external_api": {
            "primary": {
                "provider": "openai",
                "api_key": "sk-live-c-1111111111",
                "api_key_env": "OPENAI_API_KEY",
            }
        },
    }
    assert compute_config_hash(config_c) != hash_a

    run_dir = tmp_path / "artifacts" / "runs" / "run_abc123"
    persisted = persist_effective_config_artifacts(run_dir=run_dir, effective_config=config_a)
    config_path = persisted["effective_config_path"]
    hash_path = persisted["config_hash_path"]
    assert config_path is not None and hash_path is not None
    written = config_path.read_text(encoding="utf-8")
    assert "sk-live-a-1234567890" not in written
    assert "[REDACTED]" in written
    assert hash_path.read_text(encoding="utf-8").strip() == hash_a


def test_local_logger_writes_run_and_subset_jsonl(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "runs" / "run_local"
    logger = LocalJsonlLogger(run_dir)
    context = RequiredLogContext(
        run_id="run_local",
        subset_idx=2,
        phase="score",
        config_hash="hash_local",
    )

    logger.log_event(context=context, event_type="phase_started", status="ok")
    logger.log_failure(
        context=context,
        failure_type="qe_row_error",
        status="failed",
        row_id="row_001",
        error="token=super-secret-token-value",
    )

    run_events = run_dir / "events.jsonl"
    subset_events = run_dir / "subsets" / "subset_002" / "events.jsonl"
    run_failures = run_dir / "failures.jsonl"
    subset_failures = run_dir / "subsets" / "subset_002" / "failures.jsonl"

    assert run_events.exists()
    assert subset_events.exists()
    assert run_failures.exists()
    assert subset_failures.exists()

    run_event_line = run_events.read_text(encoding="utf-8").strip().splitlines()[0]
    run_failure_line = run_failures.read_text(encoding="utf-8").strip().splitlines()[0]
    run_event = json.loads(run_event_line)
    run_failure = json.loads(run_failure_line)

    assert run_event["run_id"] == "run_local"
    assert run_event["subset_idx"] == 2
    assert run_failure["row_id"] == "row_001"
    assert "[REDACTED]" in run_failure["error"]


def test_append_jsonl_record_works_without_fcntl(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(local_logging_module, "fcntl", None)
    out_path = tmp_path / "events.jsonl"
    local_logging_module.append_jsonl_record(out_path, {"hello": "world"})
    assert out_path.exists()
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["hello"] == "world"
