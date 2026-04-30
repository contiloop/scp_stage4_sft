from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scp_stage4.pipeline.data_source_ratio import main  # noqa: E402


def test_data_source_ratio_reports_counts_and_ratios(tmp_path: Path, capsys) -> None:
    artifacts_dir = tmp_path / "artifacts" / "data"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": "1", "dataset": "a"},
        {"id": "2", "dataset": "a"},
        {"id": "3", "dataset": "b"},
    ]
    path = artifacts_dir / "datapool.train.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    code = main(["--artifacts-dir", str(artifacts_dir), "--files", "datapool.train.jsonl"])
    out = capsys.readouterr().out

    assert code == 0
    assert "datapool.train.jsonl\ttotal=3" in out
    assert "a\t2\t66.67%" in out
    assert "b\t1\t33.33%" in out


def test_data_source_ratio_returns_nonzero_when_files_missing(tmp_path: Path, capsys) -> None:
    artifacts_dir = tmp_path / "artifacts" / "data"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    code = main(["--artifacts-dir", str(artifacts_dir), "--files", "missing.jsonl"])
    err = capsys.readouterr().err
    assert code == 1
    assert "No artifact files found" in err
