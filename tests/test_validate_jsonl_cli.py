from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scp_stage4.schema.validate_jsonl import main


def test_validate_jsonl_cli_passes_with_repo_fixtures() -> None:
    rc = main(["--config", str(ROOT / "configs" / "scp_stage4.yaml"), "--run-id", "local_contract"])
    assert rc == 0


def test_validate_jsonl_cli_fails_when_no_jsonl_found(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    rc = main(["--config", str(ROOT / "configs" / "scp_stage4.yaml"), "--run-id", "no_artifacts"])
    assert rc == 1
