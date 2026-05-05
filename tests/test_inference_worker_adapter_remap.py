from __future__ import annotations

import pytest

pytest.importorskip("torch")

from scp_stage4.pipeline.workers.inference_worker import (
    _has_qwen35_language_model_prefix,
    _remap_qwen35_adapter_state_dict,
)


def test_remap_qwen35_adapter_state_dict_rewrites_nested_model_prefixes() -> None:
    state_dict = {
        "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight": 1,
        "base_model.model.model.model.layers.1.mlp.down_proj.lora_B.weight": 2,
        "base_model.model.model.embed_tokens.lora_embedding_A.weight": 3,
        "base_model.model.model.norm.lora_A.weight": 4,
    }
    remapped, changed = _remap_qwen35_adapter_state_dict(state_dict)

    assert changed == 4
    assert (
        "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.weight"
        in remapped
    )
    assert (
        "base_model.model.model.model.language_model.layers.1.mlp.down_proj.lora_B.weight"
        in remapped
    )
    assert "base_model.model.model.language_model.embed_tokens.lora_embedding_A.weight" in remapped
    assert "base_model.model.model.language_model.norm.lora_A.weight" in remapped


def test_remap_qwen35_adapter_state_dict_keeps_already_remapped_keys() -> None:
    key = "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
    remapped, changed = _remap_qwen35_adapter_state_dict({key: 1})
    assert changed == 0
    assert key in remapped


def test_has_qwen35_language_model_prefix_detects_nested_parameter_names() -> None:
    class _DummyModel:
        def named_parameters(self):
            yield (
                "base_model.model.model.language_model.layers.0.self_attn.q_proj.weight",
                object(),
            )

    assert _has_qwen35_language_model_prefix(_DummyModel()) is True
