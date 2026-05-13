from __future__ import annotations

from scp_stage4.pipeline.workers.common import WorkerContractError
from scp_stage4.pipeline.workers.training_worker import (
    _filter_training_text_indices_by_length,
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
