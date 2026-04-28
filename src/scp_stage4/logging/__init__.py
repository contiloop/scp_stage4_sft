"""Local logging contracts for SCP Stage 4."""

from .local import LocalJsonlLogger, append_jsonl_record
from .schema import (
    LOG_STATUS_VALUES,
    REQUIRED_CONTEXT_FIELDS,
    LogStatus,
    RequiredLogContext,
    build_event_record,
    build_failure_record,
    build_metrics_record,
    sanitize_for_log,
    sanitize_string,
    with_required_context,
)

__all__ = [
    "LOG_STATUS_VALUES",
    "REQUIRED_CONTEXT_FIELDS",
    "LocalJsonlLogger",
    "LogStatus",
    "RequiredLogContext",
    "append_jsonl_record",
    "build_event_record",
    "build_failure_record",
    "build_metrics_record",
    "sanitize_for_log",
    "sanitize_string",
    "with_required_context",
]
