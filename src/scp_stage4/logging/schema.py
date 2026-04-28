"""Schema helpers for local JSONL logging."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import re
from typing import Any, Mapping

REDACTED = "[REDACTED]"
REQUIRED_CONTEXT_FIELDS = ("run_id", "subset_idx", "phase", "config_hash")

_OPENAI_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-~+/]+=*\b")
_KV_SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|passphrase)\b\s*[:=]\s*([^\s,;]+)"
)
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


class LogStatus(str, Enum):
    """Allowed status values across events, metrics, and failures."""

    OK = "ok"
    SKIPPED = "skipped"
    FILTERED = "filtered"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


LOG_STATUS_VALUES = {status.value for status in LogStatus}


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sanitize_string(value: Any, *, max_len: int = 4000) -> str:
    """Sanitize text to avoid control chars and accidental secret exposure."""
    text = str(value)
    text = _CONTROL_CHAR_PATTERN.sub(" ", text)
    text = _KV_SECRET_PATTERN.sub(lambda m: f"{m.group(1)}={REDACTED}", text)
    text = _BEARER_PATTERN.sub("Bearer [REDACTED]", text)
    text = _OPENAI_KEY_PATTERN.sub(REDACTED, text)
    if len(text) > max_len:
        return f"{text[:max_len]}...<truncated>"
    return text


def sanitize_for_log(value: Any) -> Any:
    """Recursively sanitize values before writing JSONL records."""
    if isinstance(value, str):
        return sanitize_string(value)
    if isinstance(value, Mapping):
        return {str(k): sanitize_for_log(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_log(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_log(item) for item in value]
    return value


def _require_non_empty_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required and must be a non-empty string")
    return sanitize_string(value.strip())


def _require_subset_idx(value: Any) -> int:
    if not isinstance(value, int):
        raise ValueError("subset_idx is required and must be an integer")
    if value < 0:
        raise ValueError("subset_idx must be >= 0")
    return value


def _validate_status(status: str) -> str:
    clean = _require_non_empty_str("status", status)
    if clean not in LOG_STATUS_VALUES:
        allowed = ", ".join(sorted(LOG_STATUS_VALUES))
        raise ValueError(f"status must be one of: {allowed}")
    return clean


def _validate_metrics_payload(metrics: Mapping[str, Any]) -> dict[str, float | int | bool]:
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError("metrics must be a non-empty mapping")
    clean_metrics: dict[str, float | int | bool] = {}
    for key, value in metrics.items():
        metric_name = _require_non_empty_str("metric key", key)
        if isinstance(value, bool):
            clean_metrics[metric_name] = value
            continue
        if not isinstance(value, (int, float)):
            raise ValueError(f"metric '{metric_name}' must be int/float/bool")
        clean_metrics[metric_name] = value
    return clean_metrics


@dataclass(frozen=True)
class RequiredLogContext:
    """Required context fields for every log record."""

    run_id: str
    subset_idx: int
    phase: str
    config_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_non_empty_str("run_id", self.run_id))
        object.__setattr__(self, "subset_idx", _require_subset_idx(self.subset_idx))
        object.__setattr__(self, "phase", _require_non_empty_str("phase", self.phase))
        object.__setattr__(self, "config_hash", _require_non_empty_str("config_hash", self.config_hash))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RequiredLogContext":
        return cls(
            run_id=value.get("run_id"),
            subset_idx=value.get("subset_idx"),
            phase=value.get("phase"),
            config_hash=value.get("config_hash"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "subset_idx": self.subset_idx,
            "phase": self.phase,
            "config_hash": self.config_hash,
        }


def with_required_context(payload: Mapping[str, Any], context: RequiredLogContext) -> dict[str, Any]:
    """Attach required context and timestamp to a payload."""
    record = dict(payload)
    record.update(context.to_dict())
    record.setdefault("timestamp", _utc_now_iso())
    return sanitize_for_log(record)


def build_event_record(
    *,
    context: RequiredLogContext,
    event_type: str,
    status: str,
    metrics: Mapping[str, Any] | None = None,
    artifact_path: str | None = None,
    error: Any = None,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a validated event record."""
    payload: dict[str, Any] = {
        "event_type": _require_non_empty_str("event_type", event_type),
        "status": _validate_status(status),
    }
    if metrics is not None:
        payload["metrics"] = _validate_metrics_payload(metrics)
    if artifact_path is not None:
        payload["artifact_path"] = _require_non_empty_str("artifact_path", artifact_path)
    payload["error"] = sanitize_for_log(error) if error is not None else None
    if extras:
        payload.update(sanitize_for_log(dict(extras)))
    return with_required_context(payload, context)


def build_metrics_record(
    *,
    context: RequiredLogContext,
    metrics: Mapping[str, Any],
    status: str = LogStatus.OK.value,
    metric_group: str | None = None,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a validated metrics record."""
    payload: dict[str, Any] = {
        "status": _validate_status(status),
        "metrics": _validate_metrics_payload(metrics),
    }
    if metric_group is not None:
        payload["metric_group"] = _require_non_empty_str("metric_group", metric_group)
    if extras:
        payload.update(sanitize_for_log(dict(extras)))
    return with_required_context(payload, context)


def build_failure_record(
    *,
    context: RequiredLogContext,
    failure_type: str,
    status: str = LogStatus.FAILED.value,
    error: Any = None,
    row_id: str | None = None,
    request_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    attempt: int | None = None,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a validated failure record."""
    payload: dict[str, Any] = {
        "failure_type": _require_non_empty_str("failure_type", failure_type),
        "status": _validate_status(status),
        "error": sanitize_for_log(error) if error is not None else None,
    }
    if row_id is not None:
        payload["row_id"] = _require_non_empty_str("row_id", row_id)
    if request_id is not None:
        payload["request_id"] = _require_non_empty_str("request_id", request_id)
    if provider is not None:
        payload["provider"] = _require_non_empty_str("provider", provider)
    if model is not None:
        payload["model"] = _require_non_empty_str("model", model)
    if attempt is not None:
        if not isinstance(attempt, int) or attempt < 0:
            raise ValueError("attempt must be an integer >= 0")
        payload["attempt"] = attempt
    if extras:
        payload.update(sanitize_for_log(dict(extras)))
    return with_required_context(payload, context)
