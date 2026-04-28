from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scp_stage4.schema import QeIsolationRequest, QeIsolationResponse, SchemaValidationError


def test_qe_isolation_request_schema_pass() -> None:
    row = {
        "id": "run_abc123/subsets/subset_000/sample_000001/q1",
        "run_id": "run_abc123",
        "subset_idx": 0,
        "row_id": "sample_000001",
        "q_tag": "q1",
        "backend": "metricx24",
        "phase": "infer-q1",
        "src": "English source text",
        "mt": "Korean candidate text",
    }
    req = QeIsolationRequest.from_dict(row)
    assert req.row_id == "sample_000001"
    assert req.subset_idx == 0


def test_qe_isolation_request_schema_fail_on_missing_required() -> None:
    row = {
        "id": "id_1",
        "q_tag": "q1",
        "backend": "metricx24",
        "src": "English source text",
        "mt": "Korean candidate text",
    }
    with pytest.raises(SchemaValidationError):
        QeIsolationRequest.from_dict(row)


def test_qe_isolation_response_schema_pass() -> None:
    row = {
        "id": "run_abc123/subsets/subset_000/sample_000001/q1",
        "score": 3.41,
        "backend": "metricx24",
        "model_name": "google/metricx-24-hybrid-xxl-v2p6-bfloat16",
        "runtime_ms": 12.4,
        "status": "ok",
    }
    resp = QeIsolationResponse.from_dict(row)
    assert resp.status == "ok"


def test_qe_isolation_response_schema_fail_on_invalid_status() -> None:
    row = {
        "id": "id_1",
        "score": 1.0,
        "backend": "metricx24",
        "model_name": "m",
        "status": "skipped",
    }
    with pytest.raises(SchemaValidationError):
        QeIsolationResponse.from_dict(row)
