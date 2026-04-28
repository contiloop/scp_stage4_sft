from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .errors import SchemaValidationError

_QE_OUTPUT_STATUS_VALUES = {"ok", "failed"}


def _require_str(data: Mapping[str, Any], key: str, *, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value.strip() == "":
        raise SchemaValidationError(f"{context}.{key} must be a non-empty string")
    return value


def _optional_str(data: Mapping[str, Any], key: str, *, context: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SchemaValidationError(f"{context}.{key} must be a string or null")
    return value


def _optional_int(data: Mapping[str, Any], key: str, *, context: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaValidationError(f"{context}.{key} must be an int or null")
    return value


def _optional_float(data: Mapping[str, Any], key: str, *, context: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{context}.{key} must be a number or null")
    return float(value)


def _require_float(data: Mapping[str, Any], key: str, *, context: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{context}.{key} must be a number")
    return float(value)


def _reject_extra_keys(data: Mapping[str, Any], *, allowed: set[str], context: str) -> None:
    extra = sorted(set(data.keys()) - allowed)
    if extra:
        raise SchemaValidationError(f"{context} has unexpected keys: {extra}")


@dataclass(frozen=True)
class QeIsolationRequest:
    id: str
    row_id: str
    q_tag: str
    backend: str
    src: str
    mt: str
    run_id: str | None = None
    subset_idx: int | None = None
    phase: str | None = None
    ref: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QeIsolationRequest":
        _reject_extra_keys(
            data,
            allowed={
                "id",
                "row_id",
                "q_tag",
                "backend",
                "src",
                "mt",
                "run_id",
                "subset_idx",
                "phase",
                "ref",
            },
            context="qe_isolation_request",
        )
        subset_idx = _optional_int(data, "subset_idx", context="qe_isolation_request")
        if subset_idx is not None and subset_idx < 0:
            raise SchemaValidationError("qe_isolation_request.subset_idx must be >= 0")
        return cls(
            id=_require_str(data, "id", context="qe_isolation_request"),
            row_id=_require_str(data, "row_id", context="qe_isolation_request"),
            q_tag=_require_str(data, "q_tag", context="qe_isolation_request"),
            backend=_require_str(data, "backend", context="qe_isolation_request"),
            src=_require_str(data, "src", context="qe_isolation_request"),
            mt=_require_str(data, "mt", context="qe_isolation_request"),
            run_id=_optional_str(data, "run_id", context="qe_isolation_request"),
            subset_idx=subset_idx,
            phase=_optional_str(data, "phase", context="qe_isolation_request"),
            ref=_optional_str(data, "ref", context="qe_isolation_request"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "row_id": self.row_id,
            "q_tag": self.q_tag,
            "backend": self.backend,
            "src": self.src,
            "mt": self.mt,
            "run_id": self.run_id,
            "subset_idx": self.subset_idx,
            "phase": self.phase,
            "ref": self.ref,
        }


@dataclass(frozen=True)
class QeIsolationResponse:
    id: str
    score: float
    backend: str
    model_name: str
    runtime_ms: float | None = None
    error: str | None = None
    status: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QeIsolationResponse":
        _reject_extra_keys(
            data,
            allowed={
                "id",
                "score",
                "backend",
                "model_name",
                "runtime_ms",
                "error",
                "status",
            },
            context="qe_isolation_response",
        )
        status = _optional_str(data, "status", context="qe_isolation_response")
        if status is not None and status not in _QE_OUTPUT_STATUS_VALUES:
            raise SchemaValidationError(
                "qe_isolation_response.status must be one of: ok, failed"
            )
        return cls(
            id=_require_str(data, "id", context="qe_isolation_response"),
            score=_require_float(data, "score", context="qe_isolation_response"),
            backend=_require_str(data, "backend", context="qe_isolation_response"),
            model_name=_require_str(data, "model_name", context="qe_isolation_response"),
            runtime_ms=_optional_float(data, "runtime_ms", context="qe_isolation_response"),
            error=_optional_str(data, "error", context="qe_isolation_response"),
            status=status,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "score": self.score,
            "backend": self.backend,
            "model_name": self.model_name,
            "runtime_ms": self.runtime_ms,
            "error": self.error,
            "status": self.status,
        }
