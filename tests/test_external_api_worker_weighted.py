"""Integration tests for the multi-provider external_api worker.

Uses monkeypatch to stub provider SDKs so no real network calls happen.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scp_stage4.pipeline.workers import external_api_worker  # noqa: E402


def _build_request(
    *,
    request_id: str,
    provider: str,
    model: str,
    split_name: str,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": request_id,
        "row_id": request_id,
        "dataset": "test-set",
        "source": "Source text.",
        "metadata": {
            "title": "t",
            "document_type": "article",
            "text_role": "body",
            "original_id": "o",
            "parent_id": None,
            "chunk_idx": None,
        },
        "request_id": f"run-1/subsets/subset_000/{request_id}/api",
        "run_id": "run-1",
        "subset_idx": 0,
        "student": "기존 번역",
        "selection": {
            "score_s": 0.0,
            "qe_q1": 24.0,
            "qe_q2": 24.0,
            "delta_qe": 0.0,
            "collapse_term": 0.0,
        },
        "prompt_version": "teacher_correction_v1",
        "prompt_hash": "abc123",
        "provider": provider,
        "model": model,
        "split_name": split_name,
        "model_params": model_params or {},
        "status": "ok",
        "config_hash": "cfg_hash",
        "runtime_config": {
            "external_api": {
                "providers": {
                    "openai":    {"api_key_env": "OPENAI_API_KEY"},
                    "anthropic": {"api_key_env": "ANTHROPIC_API_KEY"},
                    "gemini":    {"api_key_env": "GEMINI_API_KEY"},
                },
                "concurrency": {
                    "max_workers": 4,
                    "per_provider": {"openai": 2, "anthropic": 2, "gemini": 2},
                    "min_request_interval_sec": 0.0,
                },
                "timeouts": {"openai_sec": 30, "anthropic_sec": 30, "gemini_sec": 30},
            },
            "prompts": {},
        },
    }


def _stub_call_result(label: str, *, thinking: str = "") -> dict[str, Any]:
    return {
        "status": "ok",
        "gold": f"한국어 번역 결과: {label}",
        "teacher_label": "minor_edit",
        "thinking_text": thinking,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "reasoning_tokens": 25 if thinking else 0,
        },
        "latency_ms": 12.5,
        "reason": None,
        "error": None,
    }


@pytest.fixture(autouse=True)
def _stub_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_openai(row: dict[str, Any]) -> dict[str, Any]:
        return _stub_call_result(label=f"openai:{row['model']}")

    def fake_anthropic(row: dict[str, Any]) -> dict[str, Any]:
        return _stub_call_result(label=f"anthropic:{row['model']}", thinking="brief plan")

    def fake_gemini(row: dict[str, Any]) -> dict[str, Any]:
        thinking_mode = row.get("model_params", {}).get("thinking_mode", "off")
        thinking = "g-thought" if thinking_mode == "dynamic" else ""
        return _stub_call_result(label=f"gemini:{row['model']}", thinking=thinking)

    monkeypatch.setattr(external_api_worker, "_openai_call", fake_openai)
    monkeypatch.setattr(external_api_worker, "_anthropic_call", fake_anthropic)
    monkeypatch.setattr(external_api_worker, "_gemini_call", fake_gemini)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def test_worker_dispatches_each_provider(tmp_path: Path) -> None:
    rows = [
        _build_request(
            request_id="r-1",
            provider="gemini",
            model="gemini-3.1-flash-lite",
            split_name="gemini-flash-lite-off",
            model_params={"thinking_mode": "off"},
        ),
        _build_request(
            request_id="r-2",
            provider="gemini",
            model="gemini-3.1-flash-lite",
            split_name="gemini-flash-lite-dynamic",
            model_params={"thinking_mode": "dynamic"},
        ),
        _build_request(
            request_id="r-3",
            provider="openai",
            model="gpt-5.5",
            split_name="gpt-5.5-thinking",
            model_params={"reasoning_effort": "medium"},
        ),
        _build_request(
            request_id="r-4",
            provider="anthropic",
            model="claude-opus-4-7",
            split_name="opus-4.7-adaptive",
            model_params={"thinking_mode": "adaptive", "adaptive_effort": "medium"},
        ),
    ]
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    _write_jsonl(in_path, rows)

    rc = external_api_worker.main(
        [
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "--section",
            "external_api",
            "--phase",
            "call-api",
        ]
    )
    assert rc == 0

    responses = _read_jsonl(out_path)
    assert len(responses) == 4

    by_id = {r["request_id"]: r for r in responses}

    # Order should be preserved within input order.
    expected_split_for = {
        "r-1": "gemini-flash-lite-off",
        "r-2": "gemini-flash-lite-dynamic",
        "r-3": "gpt-5.5-thinking",
        "r-4": "opus-4.7-adaptive",
    }
    for row_id, split in expected_split_for.items():
        req_id = f"run-1/subsets/subset_000/{row_id}/api"
        response = by_id[req_id]
        assert response["status"] == "ok"
        assert response["split_name"] == split
        assert response["usage"]["reasoning_tokens"] >= 0

    # Adaptive Claude should have non-empty thinking_text via the stub.
    claude_resp = by_id["run-1/subsets/subset_000/r-4/api"]
    assert claude_resp["thinking_text"] == "brief plan"

    # Gemini dynamic should also surface thinking.
    gemini_dynamic = by_id["run-1/subsets/subset_000/r-2/api"]
    assert gemini_dynamic["thinking_text"] == "g-thought"

    # Gemini off should have empty thinking.
    gemini_off = by_id["run-1/subsets/subset_000/r-1/api"]
    assert gemini_off["thinking_text"] == ""


def test_worker_records_failure_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_failing(row: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(external_api_worker, "_gemini_call", fake_failing)

    rows = [
        _build_request(
            request_id="r-fail",
            provider="gemini",
            model="gemini-3.1-flash-lite",
            split_name="gemini-flash-lite-off",
        )
    ]
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    _write_jsonl(in_path, rows)

    rc = external_api_worker.main(
        [
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "--section",
            "external_api",
            "--phase",
            "call-api",
        ]
    )
    assert rc == 0

    responses = _read_jsonl(out_path)
    assert len(responses) == 1
    response = responses[0]
    assert response["status"] == "failed"
    assert response["teacher_label"] == "runtime_error"
    assert "simulated failure" in (response["error"] or "")
    # split_name still echoed back on failure.
    assert response["split_name"] == "gemini-flash-lite-off"
    assert response["usage"]["reasoning_tokens"] == 0
