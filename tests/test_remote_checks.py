from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scp_stage4.pipeline import remote_checks
from scp_stage4.pipeline.prepare_data import run_prepare_data


CONFIG_PATH = str(ROOT / "configs" / "scp_stage4.yaml")


def test_validate_env_returns_zero_without_api_key() -> None:
    rc = remote_checks.cmd_validate_env(CONFIG_PATH, [])
    assert rc == 0


def test_smoke_qe_env_mode_success(monkeypatch) -> None:
    monkeypatch.setenv("COMET_PYTHON", sys.executable)
    monkeypatch.setenv("METRICX_PYTHON", sys.executable)
    rc = remote_checks.cmd_smoke_qe(CONFIG_PATH, [])
    assert rc == 0


def test_smoke_qe_path_mode_success() -> None:
    rc = remote_checks.cmd_smoke_qe(
        CONFIG_PATH,
        [
            f"qe.isolation.env.comet_python_env={sys.executable}",
            f"qe.isolation.env.metricx_python_env={sys.executable}",
        ],
    )
    assert rc == 0


def test_smoke_model_success() -> None:
    rc = remote_checks.cmd_smoke_model(CONFIG_PATH, [])
    assert rc == 0


def test_smoke_api_placeholder_fails() -> None:
    rc = remote_checks.cmd_smoke_api(CONFIG_PATH, [])
    assert rc == 1


def test_smoke_api_non_placeholder_with_key_succeeds(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    rc = remote_checks.cmd_smoke_api(
        CONFIG_PATH,
        ["external_api.primary.model=openai/gpt-real"],
    )
    assert rc == 0


def test_dry_run_subset_generates_mock_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_prepare_data(CONFIG_PATH)
    rc = remote_checks.cmd_dry_run_subset(CONFIG_PATH, [])
    assert rc == 0
    run_root = tmp_path / "artifacts" / "runs" / "dry_run_remote_subset"
    assert (run_root / "subsets" / "subset_000" / "q1.jsonl").exists()


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["validate-env", "--config", CONFIG_PATH], 0),
        (["smoke-model", "--config", CONFIG_PATH], 0),
    ],
)
def test_remote_checks_main_basic(argv: list[str], expected: int) -> None:
    assert remote_checks.main(argv) == expected
