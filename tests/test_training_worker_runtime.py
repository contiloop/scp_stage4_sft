from __future__ import annotations

import json
from pathlib import Path

import pytest

from scp_stage4.pipeline.workers.common import WorkerContractError
from scp_stage4.pipeline.workers.training_worker import (
    _filter_training_text_indices_by_length,
    _assert_full_weight_checkpoint_keys,
    _normalize_full_weight_checkpoint_keys,
    _resolve_attention_impl,
    _resolve_response_template,
    _resolve_train_runtime,
)


def test_resolve_train_runtime_defaults_load_in_4bit_false() -> None:
    runtime = _resolve_train_runtime(
        {
            "model": {
                "name": "alwaysgood/qwen35-it",
                "max_length": 8192,
            }
        }
    )
    assert runtime.load_in_4bit is False


def test_resolve_train_runtime_respects_explicit_load_in_4bit_true() -> None:
    runtime = _resolve_train_runtime(
        {
            "model": {
                "name": "alwaysgood/qwen35-it",
                "max_length": 8192,
                "load_in_4bit": True,
            }
        }
    )
    assert runtime.load_in_4bit is True


def test_resolve_train_runtime_requires_existing_base_checkpoint_when_marked(tmp_path: Path) -> None:
    missing = tmp_path / "missing_checkpoint"
    with pytest.raises(WorkerContractError, match="required base_checkpoint path not found"):
        _resolve_train_runtime(
            {
                "model": {
                    "name": "alwaysgood/qwen35-it",
                    "max_length": 8192,
                },
                "base_checkpoint": str(missing),
                "requires_base_checkpoint": True,
            }
        )


def test_resolve_train_runtime_uses_required_base_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    runtime = _resolve_train_runtime(
        {
            "model": {
                "name": "alwaysgood/qwen35-it",
                "max_length": 8192,
            },
            "base_checkpoint": str(checkpoint),
            "requires_base_checkpoint": True,
        }
    )
    assert runtime.model_ref == str(checkpoint)


def test_resolve_train_runtime_reads_attention_impl(monkeypatch) -> None:
    monkeypatch.delenv("ATTN_IMPLEMENTATION", raising=False)
    runtime = _resolve_train_runtime(
        {
            "model": {
                "name": "alwaysgood/qwen35-it",
                "max_length": 8192,
                "attention_impl": "flash_attention_2",
            }
        }
    )
    assert runtime.attention_impl == "flash_attention_2"
    assert _resolve_attention_impl(runtime) == "flash_attention_2"


def test_resolve_attention_impl_env_overrides_config(monkeypatch) -> None:
    runtime = _resolve_train_runtime(
        {
            "model": {
                "name": "alwaysgood/qwen35-it",
                "max_length": 8192,
                "attention_impl": "flash_attention_2",
            }
        }
    )
    monkeypatch.setenv("ATTN_IMPLEMENTATION", "sdpa")
    assert _resolve_attention_impl(runtime) == "sdpa"


def test_resolve_attention_impl_defaults_to_sdpa(monkeypatch) -> None:
    runtime = _resolve_train_runtime(
        {
            "model": {
                "name": "alwaysgood/qwen35-it",
                "max_length": 8192,
            }
        }
    )
    monkeypatch.delenv("ATTN_IMPLEMENTATION", raising=False)
    assert _resolve_attention_impl(runtime) == "sdpa"


def test_resolve_response_template_from_batching() -> None:
    value = _resolve_response_template(
        {"batching": {"response_template": "### Answer:\n"}},
        phase="update-base",
    )
    assert value == "### Answer:\n"


def test_resolve_response_template_from_top_level() -> None:
    value = _resolve_response_template(
        {"response_template": "### Final:\n"},
        phase="train-collapse-lora",
    )
    assert value == "### Final:\n"


def test_resolve_response_template_raises_when_missing() -> None:
    try:
        _resolve_response_template({}, phase="update-base")
    except WorkerContractError:
        return
    raise AssertionError("expected WorkerContractError for missing response_template")


def test_resolve_response_template_from_runtime_prompts() -> None:
    value = _resolve_response_template(
        {"runtime_prompts": {"sft": {"response_template": "### YAML:\n"}}},
        phase="update-base",
    )
    assert value == "### YAML:\n"


def test_filter_training_text_indices_by_length_passes_within_limit() -> None:
    class _Tokenizer:
        def __call__(self, texts: list[str], **kwargs: object) -> dict[str, list[int]]:
            return {"length": [len(text) for text in texts]}

    keep_indices, over_limit = _filter_training_text_indices_by_length(
        tokenizer=_Tokenizer(),
        texts=["short", "also-short"],
        row_ids=["row_1", "row_2"],
        max_seq_length=32,
    )
    assert keep_indices == [0, 1]
    assert over_limit == []


def test_filter_training_text_indices_by_length_filters_overflow() -> None:
    class _Tokenizer:
        def __call__(self, texts: list[str], **kwargs: object) -> dict[str, list[int]]:
            return {"length": [len(text) for text in texts]}

    keep_indices, over_limit = _filter_training_text_indices_by_length(
        tokenizer=_Tokenizer(),
        texts=["ok", "x" * 64],
        row_ids=["row_ok", "row_over"],
        max_seq_length=32,
    )
    assert keep_indices == [0]
    assert over_limit == [("row_over", 64)]


def test_normalize_full_weight_checkpoint_keys_rewrites_unsloth_nested_prefix(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    safe_open = pytest.importorskip("safetensors").safe_open
    save_file = pytest.importorskip("safetensors.torch").save_file
    checkpoint_dir = tmp_path / "full_weight_model"
    checkpoint_dir.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    nested_key = "model.language_model.language_model.language_model.layers.0.mlp.gate_proj.weight"
    base_key = "model.language_model.layers.0.mlp.gate_proj.weight"
    save_file(
        {nested_key: torch.ones((2, 2), dtype=torch.float32)},
        str(checkpoint_dir / shard_name),
        metadata={"format": "pt"},
    )
    (checkpoint_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {nested_key: shard_name}}),
        encoding="utf-8",
    )

    assert _normalize_full_weight_checkpoint_keys(checkpoint_dir) is True
    _assert_full_weight_checkpoint_keys(checkpoint_dir)

    with safe_open(checkpoint_dir / shard_name, framework="pt") as handle:
        keys = set(handle.keys())
    assert base_key in keys
    assert nested_key not in keys
    index_payload = json.loads((checkpoint_dir / "model.safetensors.index.json").read_text())
    assert index_payload["weight_map"] == {base_key: shard_name}


def test_normalize_full_weight_checkpoint_keys_allows_nonconflicting_mixed_prefixes(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    safe_open = pytest.importorskip("safetensors").safe_open
    save_file = pytest.importorskip("safetensors.torch").save_file
    checkpoint_dir = tmp_path / "full_weight_model"
    checkpoint_dir.mkdir()
    nested_key = "model.language_model.language_model.language_model.layers.0.mlp.gate_proj.weight"
    base_key = "model.language_model.layers.1.mlp.gate_proj.weight"
    save_file(
        {
            nested_key: torch.ones((1,), dtype=torch.float32),
            base_key: torch.zeros((1,), dtype=torch.float32),
        },
        str(checkpoint_dir / "model.safetensors"),
    )

    assert _normalize_full_weight_checkpoint_keys(checkpoint_dir) is True

    with safe_open(checkpoint_dir / "model.safetensors", framework="pt") as handle:
        keys = set(handle.keys())
    assert "model.language_model.layers.0.mlp.gate_proj.weight" in keys
    assert base_key in keys
    assert nested_key not in keys
