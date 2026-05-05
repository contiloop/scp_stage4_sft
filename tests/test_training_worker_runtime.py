from __future__ import annotations

from scp_stage4.pipeline.workers.training_worker import _resolve_train_runtime


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
