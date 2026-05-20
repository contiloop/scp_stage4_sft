"""Tests for vllm_inference_worker — no vLLM dependency required."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scp_stage4.pipeline.workers.vllm_inference_worker import (
    _build_prompt,
    _build_sampling_params,
    _resolve_model_name,
    _resolve_vllm_kwargs,
    main,
)
from scp_stage4.pipeline.workers.common import WorkerContractError


def test_build_prompt_contains_source() -> None:
    prompt = _build_prompt("Hello world")
    assert "Hello world" in prompt
    assert "Korean:" in prompt


def test_resolve_model_name_requires_name() -> None:
    with pytest.raises(WorkerContractError, match="model.name is required"):
        _resolve_model_name({"runtime_config": {"model": {"name": ""}}})


def test_resolve_model_name_ok() -> None:
    name = _resolve_model_name({"runtime_config": {"model": {"name": "org/model"}}})
    assert name == "org/model"


def test_resolve_model_name_prefers_full_weight_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "full_weight_model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")

    name = _resolve_model_name(
        {
            "base_checkpoint": str(checkpoint),
            "runtime_config": {"model": {"name": "org/model"}},
        }
    )

    assert name == str(checkpoint)


def test_resolve_model_name_uses_base_model_when_checkpoint_is_lora(tmp_path: Path) -> None:
    adapter = tmp_path / "main_adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    name = _resolve_model_name(
        {
            "base_checkpoint": str(adapter),
            "runtime_config": {"model": {"name": "org/model"}},
        }
    )

    assert name == "org/model"


def test_resolve_model_name_rejects_missing_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "missing"

    with pytest.raises(WorkerContractError, match="base_checkpoint path not found"):
        _resolve_model_name(
            {
                "base_checkpoint": str(checkpoint),
                "runtime_config": {"model": {"name": "org/model"}},
            }
        )


def test_resolve_model_name_rejects_invalid_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "not_a_checkpoint"
    checkpoint.mkdir()

    with pytest.raises(WorkerContractError, match="base_checkpoint is not a full-weight checkpoint"):
        _resolve_model_name(
            {
                "base_checkpoint": str(checkpoint),
                "runtime_config": {"model": {"name": "org/model"}},
            }
        )


def test_resolve_vllm_kwargs_defaults() -> None:
    request = {"runtime_config": {"model": {"name": "m", "dtype": "bf16", "max_length": 4096}}}
    kwargs = _resolve_vllm_kwargs(request)
    assert kwargs["max_model_len"] == 4096
    assert kwargs["dtype"] == "bfloat16"
    assert kwargs["trust_remote_code"] is False


def test_resolve_vllm_kwargs_tensor_parallel() -> None:
    request = {
        "runtime_config": {
            "model": {
                "name": "m",
                "max_length": 8192,
                "tensor_parallel_size": 4,
            }
        }
    }
    kwargs = _resolve_vllm_kwargs(request)
    assert kwargs["tensor_parallel_size"] == 4


def test_resolve_vllm_kwargs_gpu_memory_utilization() -> None:
    request = {
        "runtime_config": {
            "model": {"name": "m", "max_length": 8192, "gpu_memory_utilization": 0.85}
        }
    }
    kwargs = _resolve_vllm_kwargs(request)
    assert kwargs["gpu_memory_utilization"] == 0.85


@patch("scp_stage4.pipeline.workers.vllm_inference_worker.SamplingParams", create=True)
def test_build_sampling_params_greedy(mock_sp_cls: Any) -> None:
    mock_sp_cls.side_effect = lambda **kw: kw

    from importlib import reload
    import scp_stage4.pipeline.workers.vllm_inference_worker as mod

    with patch.object(mod, "_build_sampling_params") as _:
        pass

    request = {"decoding": {"max_new_tokens": 512, "do_sample": False, "temperature": 0.0}}

    sp_module = MagicMock()
    sp_module.SamplingParams = lambda **kw: kw
    with patch.dict("sys.modules", {"vllm": sp_module}):
        result = _build_sampling_params(request)
    assert result["temperature"] == 0.0
    assert result["max_tokens"] == 512


def test_build_sampling_params_sampling() -> None:
    sp_module = MagicMock()
    sp_module.SamplingParams = lambda **kw: kw
    with patch.dict("sys.modules", {"vllm": sp_module}):
        result = _build_sampling_params(
            {"decoding": {"max_new_tokens": 4096, "do_sample": True, "temperature": 1.1, "top_p": 0.95}}
        )
    assert result["temperature"] == 1.1
    assert result["top_p"] == 0.95
    assert result["max_tokens"] == 4096


def _make_request(
    req_id: str = "test-001",
    source: str = "Hello",
    q_tag: str = "q1",
    collapse_adapter: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": req_id,
        "run_id": "run-0",
        "subset_idx": 0,
        "row_id": req_id,
        "order_idx": 0,
        "q_tag": q_tag,
        "source": source,
        "decoding": {"max_new_tokens": 256, "do_sample": False, "temperature": 0.0, "top_p": None},
        "runtime_config": {"model": {"name": "test/model", "max_length": 8192, "dtype": "bf16"}},
    }
    if collapse_adapter is not None:
        row["collapse_adapter"] = collapse_adapter
    return row


def test_main_empty_input(tmp_path: Any) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text("")

    exit_code = main([
        "--input", str(input_path),
        "--output", str(output_path),
        "--section", "inference",
        "--phase", "infer-q1",
    ])
    assert exit_code == 0
    assert output_path.exists()


def test_main_generates_output(tmp_path: Any) -> None:
    """End-to-end test with mocked vLLM engine."""
    import json

    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    rows = [_make_request("r0", "Hello"), _make_request("r1", "World")]
    input_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    mock_output = MagicMock()
    mock_output.outputs = [MagicMock(text="안녕하세요")]

    mock_llm = MagicMock()
    mock_llm.generate.return_value = [mock_output, mock_output]

    mock_vllm = MagicMock()
    mock_vllm.LLM.return_value = mock_llm
    mock_vllm.SamplingParams = lambda **kw: kw
    mock_lora_mod = MagicMock()

    with patch.dict("sys.modules", {"vllm": mock_vllm, "vllm.lora.request": mock_lora_mod}):
        exit_code = main([
            "--input", str(input_path),
            "--output", str(output_path),
            "--section", "inference",
            "--phase", "infer-q1",
        ])

    assert exit_code == 0
    assert output_path.exists()

    from scp_stage4.data import read_jsonl
    results = read_jsonl(str(output_path))
    assert len(results) == 2
    assert all(r["status"] == "ok" for r in results)
    assert all(r["mt"] == "안녕하세요" for r in results)
