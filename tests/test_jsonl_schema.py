from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scp_stage4.data import (  # noqa: E402
    RowIdValidationError,
    read_jsonl,
    validate_row_id_preservation,
    write_jsonl,
)
from scp_stage4.pipeline.io_utils import (  # noqa: E402
    read_jsonl as pipeline_read_jsonl,
    write_jsonl as pipeline_write_jsonl,
)
from scp_stage4.schema import (  # noqa: E402
    SchemaValidationError,
    validate_artifact_row,
    validate_artifact_rows,
)

FIXTURES = ROOT / "tests" / "fixtures"


def _fixture_rows(name: str) -> list[dict[str, object]]:
    return read_jsonl(FIXTURES / name)  # type: ignore[return-value]


def test_schema_validation_pass() -> None:
    fixture_to_artifact = [
        ("input.happy.jsonl", "input"),
        ("q1.happy.jsonl", "q1"),
        ("q2.happy.jsonl", "q2"),
        ("scored.happy.jsonl", "scored"),
        ("selected.happy.jsonl", "selected"),
        ("api_requests.happy.jsonl", "api_requests"),
        ("api.happy.jsonl", "api"),
        ("preference_pairs.happy.jsonl", "preference_pairs"),
        ("train.happy.jsonl", "train"),
    ]

    for fixture_name, artifact_name in fixture_to_artifact:
        rows = _fixture_rows(fixture_name)
        validated = validate_artifact_rows(rows, artifact_name)  # type: ignore[arg-type]
        assert len(validated) == len(rows)


def test_schema_validation_fail() -> None:
    invalid_row = {
        "id": "sample_000001",
        "dataset": "alwaysgood/reuter_processed",
        "source": "A valid English source",
        "metadata": {
            "title": "Bad Doc Type",
            "document_type": "memo",
            "text_role": "body",
            "original_id": None,
            "parent_id": None,
            "chunk_idx": None,
        },
    }

    with pytest.raises(SchemaValidationError):
        validate_artifact_row(invalid_row, "input")

    invalid_api_request = {
        "id": "sample_000001",
        "dataset": "alwaysgood/reuter_processed",
        "source": "A valid English source",
        "metadata": {
            "title": "Revenue Beat Expectations",
            "document_type": "article",
            "text_role": "body",
            "original_id": "reuter_1",
            "parent_id": None,
            "chunk_idx": None,
        },
        "request_id": "run_abc123/subsets/subset_000/sample_000001/api",
        "student": "학생 번역",
    }
    with pytest.raises(SchemaValidationError):
        validate_artifact_row(invalid_api_request, "api_requests")


def test_jsonl_io_roundtrip(tmp_path: Path) -> None:
    input_rows = validate_artifact_rows(_fixture_rows("input.happy.jsonl"), "input")
    output_path = tmp_path / "roundtrip.jsonl"

    count = write_jsonl(output_path, input_rows)
    assert count == len(input_rows)

    raw = output_path.read_text(encoding="utf-8")
    assert raw.count("\n") == len(input_rows)

    loaded = read_jsonl(output_path)
    assert loaded == input_rows


def test_pipeline_jsonl_utf8_is_not_escaped(tmp_path: Path) -> None:
    output_path = tmp_path / "utf8_pipeline.jsonl"
    rows = [{"id": "row_1", "source": "한글 문장", "dataset": "fixture", "metadata": {"text_role": "body"}}]
    pipeline_write_jsonl(output_path, rows)
    raw = output_path.read_text(encoding="utf-8")
    assert "한글 문장" in raw
    assert "\\u" not in raw


def test_pipeline_jsonl_empty_line_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id":"row_1"}\n\n{"id":"row_2"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        pipeline_read_jsonl(path)


def test_row_id_preservation_pass() -> None:
    input_rows = _fixture_rows("input.happy.jsonl")
    q1_rows = _fixture_rows("q1.happy.jsonl")
    q2_rows = _fixture_rows("q2.happy.jsonl")
    scored_rows = _fixture_rows("scored.happy.jsonl")
    selected_rows = _fixture_rows("selected.happy.jsonl")
    api_requests_rows = _fixture_rows("api_requests.happy.jsonl")
    api_rows = _fixture_rows("api.happy.jsonl")
    preference_rows = _fixture_rows("preference_pairs.happy.jsonl")
    train_rows = _fixture_rows("train.happy.jsonl")

    validate_row_id_preservation(input_rows, q1_rows, base_name="input", candidate_name="q1")
    validate_row_id_preservation(q1_rows, q2_rows, base_name="q1", candidate_name="q2")
    validate_row_id_preservation(q2_rows, scored_rows, base_name="q2", candidate_name="scored")
    validate_row_id_preservation(
        scored_rows,
        selected_rows,
        allow_subset=True,
        base_name="scored",
        candidate_name="selected",
    )
    validate_row_id_preservation(
        selected_rows,
        api_requests_rows,
        allow_subset=True,
        base_name="selected",
        candidate_name="api_requests",
    )
    validate_row_id_preservation(
        api_requests_rows,
        api_rows,
        allow_subset=True,
        base_name="api_requests",
        candidate_name="api",
    )
    validate_row_id_preservation(
        api_rows,
        preference_rows,
        allow_subset=True,
        base_name="api",
        candidate_name="preference_pairs",
    )
    validate_row_id_preservation(
        api_rows,
        train_rows,
        allow_subset=True,
        base_name="api",
        candidate_name="train",
    )


def test_row_id_preservation_fail() -> None:
    input_rows = _fixture_rows("input.happy.jsonl")
    q1_drift_rows = _fixture_rows("q1.row_id_drift.jsonl")

    with pytest.raises(RowIdValidationError):
        validate_row_id_preservation(
            input_rows,
            q1_drift_rows,
            base_name="input",
            candidate_name="q1",
        )
