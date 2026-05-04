from __future__ import annotations

import subprocess

import pytest

from scp_stage4.pipeline.workers.qe_worker import _metricx24_scores, _resolve_qe_python
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


def test_metricx_scores_uses_local_metricx_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"src": "hello", "mt": "안녕"}]

    def _fake_run(cmd: list[str], capture_output: bool, text: bool) -> subprocess.CompletedProcess[str]:
        assert "scp_stage4.pipeline.workers.metricx_driver" in cmd
        output_path = cmd[cmd.index("--output_file") + 1]
        from scp_stage4.data import write_jsonl

        write_jsonl(output_path, [{"prediction": 3.14}], ensure_ascii=False)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("scp_stage4.pipeline.workers.qe_worker.subprocess.run", _fake_run)
    scores = _metricx24_scores(
        rows,
        python_executable="/tmp/fake/python",
        model_name="google/metricx-24-hybrid-xxl-v2p6-bfloat16",
        tokenizer_name="google/mt5-xl",
        batch_size=8,
        max_input_length=1536,
    )
    assert scores == [3.14]
