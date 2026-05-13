from __future__ import annotations

from scp_stage4.pipeline.workers.common import WorkerContractError
from scp_stage4.pipeline.workers.training_worker import (
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
