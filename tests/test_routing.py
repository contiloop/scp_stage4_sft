"""Tests for weighted routing of external API requests."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scp_stage4.pipeline.routing import (  # noqa: E402
    RoutingConfigError,
    Split,
    assign_split,
    expected_counts,
    parse_routing,
)


def _four_split_cfg() -> dict:
    return {
        "mode": "weighted",
        "seed": 42,
        "splits": [
            {
                "name": "gemini-3.1-flash-lite-off",
                "provider": "gemini",
                "model": "gemini-3.1-flash-lite",
                "params": {"thinking_mode": "off"},
                "weight": 0.7,
            },
            {
                "name": "gemini-3.1-flash-lite-dynamic",
                "provider": "gemini",
                "model": "gemini-3.1-flash-lite",
                "params": {"thinking_mode": "dynamic"},
                "weight": 0.1,
            },
            {
                "name": "gpt-5.5-thinking",
                "provider": "openai",
                "model": "gpt-5.5",
                "params": {"reasoning_effort": "medium"},
                "weight": 0.1,
            },
            {
                "name": "opus-4.7-adaptive",
                "provider": "anthropic",
                "model": "claude-opus-4-7",
                "params": {"thinking_mode": "adaptive", "adaptive_effort": "medium"},
                "weight": 0.1,
            },
        ],
    }


def test_parse_routing_single_default() -> None:
    plan = parse_routing({"mode": "single"})
    assert plan.mode == "single"
    assert plan.splits == ()
    assert not plan.is_weighted


def test_parse_routing_normalizes_weights() -> None:
    cfg = _four_split_cfg()
    # Inflate weights to verify renormalization to 1.0.
    cfg["splits"][0]["weight"] = 70
    cfg["splits"][1]["weight"] = 10
    cfg["splits"][2]["weight"] = 10
    cfg["splits"][3]["weight"] = 10
    plan = parse_routing(cfg)
    assert plan.is_weighted
    total = sum(s.weight for s in plan.splits)
    assert total == pytest.approx(1.0, abs=1e-9)


def test_parse_routing_rejects_invalid() -> None:
    with pytest.raises(RoutingConfigError):
        parse_routing({"mode": "weighted", "splits": []})

    with pytest.raises(RoutingConfigError):
        parse_routing(
            {
                "mode": "weighted",
                "splits": [
                    {
                        "name": "x",
                        "provider": "openai",
                        "model": "gpt-5.5",
                        "weight": 0,
                    }
                ],
            }
        )

    with pytest.raises(RoutingConfigError):
        parse_routing(
            {
                "mode": "weighted",
                "splits": [
                    {
                        "name": "x",
                        "provider": "fake-provider",
                        "model": "gpt-5.5",
                        "weight": 1.0,
                    }
                ],
            }
        )

    with pytest.raises(RoutingConfigError):
        parse_routing(
            {
                "mode": "weighted",
                "splits": [
                    {
                        "name": "dup",
                        "provider": "openai",
                        "model": "gpt-5.5",
                        "weight": 0.5,
                    },
                    {
                        "name": "dup",
                        "provider": "openai",
                        "model": "gpt-5.5",
                        "weight": 0.5,
                    },
                ],
            }
        )


def test_assign_split_is_deterministic() -> None:
    plan = parse_routing(_four_split_cfg())
    row_id = "doc-00042"
    first = assign_split(row_id=row_id, plan=plan)
    second = assign_split(row_id=row_id, plan=plan)
    assert first.name == second.name


def test_assign_split_distribution_converges() -> None:
    plan = parse_routing(_four_split_cfg())
    total = 20_000
    counts: Counter[str] = Counter()
    for i in range(total):
        chosen = assign_split(row_id=f"row-{i}", plan=plan)
        counts[chosen.name] += 1

    # Validate empirical ratio is within 1% of target.
    expectations = {
        "gemini-3.1-flash-lite-off": 0.70,
        "gemini-3.1-flash-lite-dynamic": 0.10,
        "gpt-5.5-thinking": 0.10,
        "opus-4.7-adaptive": 0.10,
    }
    for name, target in expectations.items():
        observed = counts[name] / total
        assert abs(observed - target) < 0.01, (
            f"split {name}: observed={observed:.4f} target={target:.2f}"
        )


def test_assign_split_distinct_seeds_diverge() -> None:
    plan_a = parse_routing(_four_split_cfg())
    cfg = _four_split_cfg()
    cfg["seed"] = 9999
    plan_b = parse_routing(cfg)
    diffs = 0
    for i in range(2000):
        a = assign_split(row_id=f"r-{i}", plan=plan_a).name
        b = assign_split(row_id=f"r-{i}", plan=plan_b).name
        if a != b:
            diffs += 1
    # Different seeds must produce notably different assignment.
    assert diffs > 200


def test_expected_counts_for_weighted_plan() -> None:
    plan = parse_routing(_four_split_cfg())
    counts = expected_counts(plan=plan, total_rows=1000)
    assert counts["gemini-3.1-flash-lite-off"] == pytest.approx(700.0)
    assert counts["opus-4.7-adaptive"] == pytest.approx(100.0)


def test_assign_split_on_single_plan_raises() -> None:
    plan = parse_routing({"mode": "single"})
    with pytest.raises(RoutingConfigError):
        assign_split(row_id="x", plan=plan)


def test_split_to_dict_preserves_params() -> None:
    split = Split(
        name="x",
        provider="openai",
        model="gpt-5.5",
        weight=1.0,
        params={"reasoning_effort": "medium"},
    )
    serialized = split.to_dict()
    assert serialized["params"] == {"reasoning_effort": "medium"}
    assert serialized["name"] == "x"
