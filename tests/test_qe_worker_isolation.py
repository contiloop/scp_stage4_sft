from __future__ import annotations

import os

import pytest

from scp_stage4.pipeline.workers.qe_worker import _resolve_qe_python
from scp_stage4.pipeline.workers.common import WorkerContractError


def _row_with_isolation() -> dict[str, object]:
    return {
        "runtime_config": {
            "qe_isolation": {
                "env": {
                    "comet_python_env": "COMET_PYTHON",
                    "metricx_python_env": "METRICX_PYTHON",
                }
            }
        }
    }


def test_metricx_backend_prefers_metricx_python_env(monkeypatch: pytest.MonkeyPatch) -> None:
    row = _row_with_isolation()
    monkeypatch.setenv("METRICX_PYTHON", "/tmp/metricx/python")
    monkeypatch.setenv("COMET_PYTHON", "/tmp/comet/python")
    assert _resolve_qe_python([row], backend="metricx24") == "/tmp/metricx/python"


def test_metricx_backend_falls_back_to_comet_python_env(monkeypatch: pytest.MonkeyPatch) -> None:
    row = _row_with_isolation()
    monkeypatch.delenv("METRICX_PYTHON", raising=False)
    monkeypatch.setenv("COMET_PYTHON", "/tmp/comet/python")
    assert _resolve_qe_python([row], backend="metricx24") == "/tmp/comet/python"


def test_metricx_backend_requires_env(monkeypatch: pytest.MonkeyPatch) -> None:
    row = _row_with_isolation()
    monkeypatch.delenv("METRICX_PYTHON", raising=False)
    monkeypatch.delenv("COMET_PYTHON", raising=False)
    with pytest.raises(WorkerContractError):
        _resolve_qe_python([row], backend="metricx24")

