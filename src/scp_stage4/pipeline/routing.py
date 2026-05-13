"""Weighted routing of external API requests to one of several model splits.

The routing module assigns each request row to exactly one ``split`` based on a
deterministic hash of ``(seed, row_id)``. Given enough rows the assignment
distribution converges to the configured weights.

Config shape (read from ``external_api.routing``)::

    routing:
      mode: "weighted"            # or "single" (legacy)
      seed: 42
      splits:
        - name: "gemini-flash-lite-off"
          provider: "gemini"
          model: "gemini-3.1-flash-lite"
          params: {thinking_mode: "off"}
          weight: 0.7
        - name: "opus-4.7-adaptive"
          provider: "anthropic"
          model: "claude-opus-4-7"
          params: {thinking_mode: "adaptive", adaptive_effort: "medium"}
          weight: 0.1
        ...

When ``mode != "weighted"`` the module returns ``None`` from ``load_splits`` and
callers fall back to the single-provider path defined under
``external_api.primary``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Mapping, Sequence


class RoutingConfigError(ValueError):
    """Raised when the routing config is invalid."""


_WEIGHT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Split:
    name: str
    provider: str
    model: str
    weight: float
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "weight": self.weight,
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class RoutingPlan:
    mode: str
    seed: int
    splits: tuple[Split, ...]

    @property
    def is_weighted(self) -> bool:
        return self.mode == "weighted" and bool(self.splits)


def _coerce_weight(raw: Any, *, name: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise RoutingConfigError(
            f"routing.splits[{name}].weight must be a number, got {raw!r}"
        ) from exc
    if value < 0:
        raise RoutingConfigError(
            f"routing.splits[{name}].weight must be >= 0, got {value}"
        )
    return value


def _coerce_params(raw: Any, *, name: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise RoutingConfigError(
            f"routing.splits[{name}].params must be a mapping, got {type(raw).__name__}"
        )
    return dict(raw)


def parse_routing(routing_cfg: Any) -> RoutingPlan:
    """Parse + validate the ``external_api.routing`` block.

    Returns a ``RoutingPlan`` with normalized weights summing to 1.0.
    """

    if routing_cfg is None:
        return RoutingPlan(mode="single", seed=0, splits=())
    if not isinstance(routing_cfg, Mapping):
        raise RoutingConfigError(
            f"external_api.routing must be a mapping, got {type(routing_cfg).__name__}"
        )

    mode = str(routing_cfg.get("mode", "single")).strip().lower() or "single"
    if mode not in {"single", "weighted"}:
        raise RoutingConfigError(
            f"external_api.routing.mode must be 'single' or 'weighted', got {mode!r}"
        )

    seed_raw = routing_cfg.get("seed", 0)
    try:
        seed = int(seed_raw)
    except (TypeError, ValueError) as exc:
        raise RoutingConfigError(
            f"external_api.routing.seed must be int, got {seed_raw!r}"
        ) from exc

    if mode == "single":
        return RoutingPlan(mode="single", seed=seed, splits=())

    splits_raw = routing_cfg.get("splits")
    if not isinstance(splits_raw, Sequence) or not splits_raw:
        raise RoutingConfigError(
            "external_api.routing.splits must be a non-empty list when mode='weighted'"
        )

    parsed: list[Split] = []
    names_seen: set[str] = set()
    for idx, entry in enumerate(splits_raw):
        if not isinstance(entry, Mapping):
            raise RoutingConfigError(
                f"external_api.routing.splits[{idx}] must be a mapping"
            )
        name = str(entry.get("name", "")).strip()
        if not name:
            raise RoutingConfigError(
                f"external_api.routing.splits[{idx}].name is required"
            )
        if name in names_seen:
            raise RoutingConfigError(
                f"external_api.routing.splits[{idx}].name={name!r} duplicates an earlier split"
            )
        names_seen.add(name)

        provider = str(entry.get("provider", "")).strip().lower()
        if provider not in {"openai", "anthropic", "gemini"}:
            raise RoutingConfigError(
                f"external_api.routing.splits[{name}].provider must be one of "
                f"openai|anthropic|gemini, got {provider!r}"
            )

        model = str(entry.get("model", "")).strip()
        if not model:
            raise RoutingConfigError(
                f"external_api.routing.splits[{name}].model is required"
            )

        weight = _coerce_weight(entry.get("weight", 0.0), name=name)
        params = _coerce_params(entry.get("params"), name=name)

        parsed.append(
            Split(name=name, provider=provider, model=model, weight=weight, params=params)
        )

    total = sum(s.weight for s in parsed)
    if total <= 0:
        raise RoutingConfigError(
            "external_api.routing.splits weights must include at least one positive value"
        )

    # Normalize to sum=1.0 so configs with weights summing to 100 (percent) also work.
    if abs(total - 1.0) > _WEIGHT_TOLERANCE:
        parsed = [
            Split(
                name=s.name,
                provider=s.provider,
                model=s.model,
                weight=s.weight / total,
                params=s.params,
            )
            for s in parsed
        ]

    return RoutingPlan(mode="weighted", seed=seed, splits=tuple(parsed))


def _uniform_unit(seed: int, row_id: str) -> float:
    payload = f"{seed}:{row_id}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def assign_split(*, row_id: str, plan: RoutingPlan) -> Split:
    """Deterministically choose a split for the given row_id.

    Uses a stable hash so re-running the pipeline routes the same row to the
    same model.
    """

    if not plan.is_weighted:
        raise RoutingConfigError("assign_split called on non-weighted plan")

    u = _uniform_unit(plan.seed, row_id)
    cumulative = 0.0
    for s in plan.splits:
        cumulative += s.weight
        if u < cumulative:
            return s
    return plan.splits[-1]


def expected_counts(*, plan: RoutingPlan, total_rows: int) -> dict[str, float]:
    """Return expected row count per split given total_rows.

    Pure helper used by reporting / sanity checks.
    """

    if not plan.is_weighted:
        return {}
    return {s.name: s.weight * total_rows for s in plan.splits}


__all__ = [
    "RoutingConfigError",
    "RoutingPlan",
    "Split",
    "assign_split",
    "expected_counts",
    "parse_routing",
]
