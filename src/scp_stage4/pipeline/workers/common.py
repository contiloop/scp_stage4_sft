"""Shared worker CLI and lightweight contract helpers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


class WorkerContractError(ValueError):
    """Raised when subprocess worker IO contract is violated."""


@dataclass(frozen=True)
class WorkerArgs:
    input_path: str
    output_path: str
    effective_config: str | None
    config_hash: str | None
    run_id: str | None
    subset_idx: str | None
    section: str | None
    phase: str | None


@dataclass(frozen=True)
class WorkerPhaseSchema:
    section: str
    phase: str
    request_required_fields: frozenset[str]
    response_required_fields: frozenset[str]

    @property
    def key(self) -> tuple[str, str]:
        return (self.section, self.phase)


_PHASE_SCHEMAS = {
    schema.key: schema
    for schema in (
        WorkerPhaseSchema(
            section="inference",
            phase="infer-q1",
            request_required_fields=frozenset(
                {"id", "run_id", "subset_idx", "row_id", "q_tag", "source", "decoding"}
            ),
            response_required_fields=frozenset({"id", "status", "mt", "error"}),
        ),
        WorkerPhaseSchema(
            section="inference",
            phase="infer-q2",
            request_required_fields=frozenset(
                {
                    "id",
                    "run_id",
                    "subset_idx",
                    "row_id",
                    "q_tag",
                    "source",
                    "decoding",
                    "collapse_adapter",
                }
            ),
            response_required_fields=frozenset({"id", "status", "mt", "error"}),
        ),
        WorkerPhaseSchema(
            section="inference",
            phase="infer-ood",
            request_required_fields=frozenset(
                {"id", "run_id", "subset_idx", "row_id", "q_tag", "source", "decoding"}
            ),
            response_required_fields=frozenset({"id", "status", "mt", "error"}),
        ),
        WorkerPhaseSchema(
            section="qe",
            phase="qe-q1",
            request_required_fields=frozenset(
                {"id", "row_id", "q_tag", "backend", "src", "mt"}
            ),
            response_required_fields=frozenset(
                {"id", "score", "backend", "model_name", "runtime_ms", "status", "error"}
            ),
        ),
        WorkerPhaseSchema(
            section="qe",
            phase="qe-q2",
            request_required_fields=frozenset(
                {"id", "row_id", "q_tag", "backend", "src", "mt"}
            ),
            response_required_fields=frozenset(
                {"id", "score", "backend", "model_name", "runtime_ms", "status", "error"}
            ),
        ),
        WorkerPhaseSchema(
            section="qe",
            phase="eval-ood",
            request_required_fields=frozenset(
                {"id", "row_id", "q_tag", "backend", "src", "mt"}
            ),
            response_required_fields=frozenset(
                {"id", "score", "backend", "model_name", "runtime_ms", "status", "error"}
            ),
        ),
        WorkerPhaseSchema(
            section="external_api",
            phase="call-api",
            request_required_fields=frozenset(
                {
                    "id",
                    "row_id",
                    "dataset",
                    "source",
                    "metadata",
                    "request_id",
                    "run_id",
                    "subset_idx",
                    "student",
                    "selection",
                    "prompt_version",
                    "prompt_hash",
                    "provider",
                    "model",
                    "status",
                    "config_hash",
                }
            ),
            response_required_fields=frozenset(
                {
                    "request_id",
                    "status",
                    "teacher_label",
                    "usage",
                    "cost",
                    "latency_ms",
                    "attempt",
                    "reason",
                    "error",
                }
            ),
        ),
        WorkerPhaseSchema(
            section="training",
            phase="train-collapse-lora",
            request_required_fields=frozenset(
                {
                    "id",
                    "run_id",
                    "subset_idx",
                    "phase",
                    "source",
                    "target",
                    "metadata",
                    "adapter_path",
                    "training_config",
                    "model",
                }
            ),
            response_required_fields=frozenset(
                {"status", "adapter_path", "trained_rows", "backend", "error"}
            ),
        ),
        WorkerPhaseSchema(
            section="training",
            phase="unload-collapse-lora",
            request_required_fields=frozenset(
                {"run_id", "subset_idx", "phase", "adapter_path"}
            ),
            response_required_fields=frozenset(
                {
                    "status",
                    "adapter_path",
                    "clean_base",
                    "active_adapters",
                    "collapse_merged",
                    "adapter_registry_hash",
                    "verified_adapter_path",
                    "backend",
                    "error",
                }
            ),
        ),
        WorkerPhaseSchema(
            section="training",
            phase="update-base",
            request_required_fields=frozenset(
                {
                    "id",
                    "run_id",
                    "subset_idx",
                    "phase",
                    "source",
                    "target",
                    "metadata",
                    "train_artifact",
                    "output_dir",
                    "training_config",
                    "model",
                }
            ),
            response_required_fields=frozenset(
                {"status", "checkpoint_path", "trained_rows", "backend", "error"}
            ),
        ),
    )
}


def resolve_phase_schema(*, section: str | None, phase: str | None) -> WorkerPhaseSchema:
    if not isinstance(section, str) or not section.strip():
        raise WorkerContractError("worker CLI missing --section")
    if not isinstance(phase, str) or not phase.strip():
        raise WorkerContractError("worker CLI missing --phase")
    key = (section, phase)
    schema = _PHASE_SCHEMAS.get(key)
    if schema is None:
        raise WorkerContractError(
            f"unsupported worker phase contract for section={section!r}, phase={phase!r}"
        )
    return schema


def parse_worker_args(
    *,
    description: str,
    argv: list[str] | None = None,
) -> WorkerArgs:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--effective-config", default=None)
    parser.add_argument("--config-hash", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--subset-idx", default=None)
    parser.add_argument("--section", default=None)
    parser.add_argument("--phase", default=None)
    parsed = parser.parse_args(argv)
    return WorkerArgs(
        input_path=parsed.input,
        output_path=parsed.output,
        effective_config=parsed.effective_config,
        config_hash=parsed.config_hash,
        run_id=parsed.run_id,
        subset_idx=parsed.subset_idx,
        section=parsed.section,
        phase=parsed.phase,
    )


def require_request_fields(
    rows: Iterable[Mapping[str, Any]],
    *,
    required_fields: set[str],
    context: str,
) -> None:
    for idx, row in enumerate(rows):
        missing = [field for field in required_fields if field not in row]
        if missing:
            raise WorkerContractError(
                f"{context} request row {idx} is missing required fields: {missing}"
            )


def validate_phase_request_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    args: WorkerArgs,
    context: str | None = None,
) -> WorkerPhaseSchema:
    schema = resolve_phase_schema(section=args.section, phase=args.phase)
    require_request_fields(
        rows,
        required_fields=set(schema.request_required_fields),
        context=context or f"{schema.section}.{schema.phase}",
    )
    return schema


def require_response_fields(
    rows: Iterable[Mapping[str, Any]],
    *,
    required_fields: set[str],
    context: str,
) -> None:
    for idx, row in enumerate(rows):
        missing = [field for field in required_fields if field not in row]
        if missing:
            raise WorkerContractError(
                f"{context} response row {idx} is missing required fields: {missing}"
            )


def validate_phase_response_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    schema: WorkerPhaseSchema,
    context: str | None = None,
) -> None:
    require_response_fields(
        rows,
        required_fields=set(schema.response_required_fields),
        context=context or f"{schema.section}.{schema.phase}",
    )
