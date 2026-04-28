from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scp_stage4.data import read_jsonl  # noqa: E402
from scp_stage4.pipeline.prepare_data import run_prepare_data  # noqa: E402
from scp_stage4.schema import validate_artifact_rows  # noqa: E402


def test_prepare_data_writes_expected_artifacts(tmp_path: Path) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd()
    try:
        # Ensure fixture lookup falls back deterministically.
        (workdir / "tests" / "fixtures").mkdir(parents=True, exist_ok=True)
        os_config = str(ROOT / "configs" / "scp_stage4.yaml")

        # Use fixed_size strategy via overrides for deterministic sample count.
        os.chdir(workdir)
        summary = run_prepare_data(
            config_path=os_config,
            overrides=[
                "pipeline.subset.strategy=fixed_size",
                "pipeline.subset.fixed_size=32",
                "data.sampling.strategy=first_n",
                "data.subset_size=32",
            ],
        )

        out_dir = workdir / "artifacts" / "data"
        expected = [
            out_dir / "datapool.normalized.jsonl",
            out_dir / "datapool.train.jsonl",
            out_dir / "datapool.eval.jsonl",
            out_dir / "datapool.train.sampled.jsonl",
            out_dir / "ood_test.jsonl",
            out_dir / "prepare_data_summary.json",
        ]
        for path in expected:
            assert path.exists(), f"missing artifact: {path}"

        normalized_rows = read_jsonl(out_dir / "datapool.normalized.jsonl")
        train_rows = read_jsonl(out_dir / "datapool.train.jsonl")
        eval_rows = read_jsonl(out_dir / "datapool.eval.jsonl")
        sampled_rows = read_jsonl(out_dir / "datapool.train.sampled.jsonl")
        validate_artifact_rows(normalized_rows, "normalized")
        validate_artifact_rows(train_rows, "normalized")
        validate_artifact_rows(eval_rows, "normalized")
        validate_artifact_rows(sampled_rows, "normalized")

        train_ids = {row["id"] for row in train_rows}
        eval_ids = {row["id"] for row in eval_rows}
        sampled_ids = [row["id"] for row in sampled_rows]

        assert train_ids.isdisjoint(eval_ids)
        assert set(sampled_ids).issubset(train_ids)
        assert len(sampled_rows) == summary["sampled_rows"]

        summary_file = json.loads((out_dir / "prepare_data_summary.json").read_text(encoding="utf-8"))
        assert summary_file["artifact_dir"].endswith("artifacts/data")
    finally:
        os.chdir(old_cwd)


def test_prepare_data_overflow_split_creates_chunk_ids(tmp_path: Path) -> None:
    workdir = tmp_path / "work_split"
    workdir.mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd()
    try:
        os.chdir(workdir)
        run_prepare_data(
            config_path=str(ROOT / "configs" / "scp_stage4.yaml"),
            overrides=[
                "data.length.max_source_tokens=3",
                "data.length.overflow=split",
                "data.length.split.max_chunks_per_row=2",
                "data.subset_size=8",
            ],
        )
        rows = read_jsonl(workdir / "artifacts" / "data" / "datapool.normalized.jsonl")
        assert any("__chunk_" in str(row["id"]) for row in rows)
    finally:
        os.chdir(old_cwd)
