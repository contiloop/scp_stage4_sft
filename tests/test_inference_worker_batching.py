from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from scp_stage4.pipeline.workers.inference_worker import (
    _BatchItem,
    _ThroughputRuntime,
    _generate_responses,
    _group_batches,
)


def _item(idx: int, prompt_tokens: int, max_new_tokens: int) -> _BatchItem:
    return _BatchItem(
        order_idx=idx,
        prompt="x" * prompt_tokens,
        prompt_tokens=prompt_tokens,
        request={
            "id": f"req-{idx}",
            "source": "hello",
            "decoding": {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "temperature": 0.0,
                "top_p": None,
            },
        },
    )


def test_group_batches_respects_token_budget() -> None:
    items = [
        _item(0, 100, 4096),
        _item(1, 100, 4096),
        _item(2, 100, 4096),
    ]
    batches = _group_batches(items=items, max_batch_tokens=9000)
    assert len(batches) == 2
    assert len(batches[0]) == 2
    assert len(batches[1]) == 1


def test_generate_responses_restores_original_order(monkeypatch: Any) -> None:
    class _Tokenizer:
        def __call__(self, prompts: list[str], **kwargs: Any) -> dict[str, Any]:
            return {"length": [len(prompt) for prompt in prompts]}

    requests = [
        {"id": "r0", "source": "A", "decoding": {"max_new_tokens": 4096}},
        {"id": "r1", "source": "AAAAA", "decoding": {"max_new_tokens": 4096}},
        {"id": "r2", "source": "AA", "decoding": {"max_new_tokens": 4096}},
    ]

    def _fake_generate_batch(**kwargs: Any) -> dict[int, str]:
        batch = kwargs["batch"]
        return {item.order_idx: f"MT::{item.request['id']}" for item in batch}

    monkeypatch.setattr(
        "scp_stage4.pipeline.workers.inference_worker._generate_batch",
        _fake_generate_batch,
    )

    responses, metrics = _generate_responses(
        model=object(),
        tokenizer=_Tokenizer(),
        requests=requests,
        throughput=_ThroughputRuntime(
            strategy="token_budget",
            max_batch_tokens=50000,
            pad_to_multiple_of=None,
            preserve_order=False,
            restore_order_in_artifacts=True,
        ),
    )
    assert [row["id"] for row in responses] == ["r0", "r1", "r2"]
    assert all(row["status"] == "ok" for row in responses)
    assert metrics["batch_count"] >= 1


def test_generate_responses_splits_batch_on_oom(monkeypatch: Any) -> None:
    class _Tokenizer:
        def __call__(self, prompts: list[str], **kwargs: Any) -> dict[str, Any]:
            return {"length": [len(prompt) for prompt in prompts]}

    requests = [
        {"id": "r0", "source": "A", "decoding": {"max_new_tokens": 4096}},
        {"id": "r1", "source": "B", "decoding": {"max_new_tokens": 4096}},
        {"id": "r2", "source": "C", "decoding": {"max_new_tokens": 4096}},
        {"id": "r3", "source": "D", "decoding": {"max_new_tokens": 4096}},
    ]
    state = {"oom_raised": False}

    def _fake_generate_batch(**kwargs: Any) -> dict[int, str]:
        batch = kwargs["batch"]
        if len(batch) >= 4 and not state["oom_raised"]:
            state["oom_raised"] = True
            raise torch.cuda.OutOfMemoryError("forced oom")
        return {item.order_idx: f"MT::{item.request['id']}" for item in batch}

    monkeypatch.setattr(
        "scp_stage4.pipeline.workers.inference_worker._generate_batch",
        _fake_generate_batch,
    )

    responses, metrics = _generate_responses(
        model=object(),
        tokenizer=_Tokenizer(),
        requests=requests,
        throughput=_ThroughputRuntime(
            strategy="token_budget",
            max_batch_tokens=9999999,
            pad_to_multiple_of=None,
            preserve_order=True,
            restore_order_in_artifacts=True,
        ),
    )

    assert all(row["status"] == "ok" for row in responses)
    assert metrics["oom_retry_count"] >= 1
