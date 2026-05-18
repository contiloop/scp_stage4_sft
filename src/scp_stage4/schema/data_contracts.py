from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, TypeAlias

from .errors import SchemaValidationError

DocumentType: TypeAlias = Literal["article", "filing", "earnings_call", "other"]
TextRole: TypeAlias = Literal["title", "body", "section", "other"]
ArtifactName: TypeAlias = Literal[
    "normalized",
    "input",
    "q1",
    "q2",
    "scored",
    "selected",
    "api_requests",
    "api",
    "preference_pairs",
    "train",
]
StatusValue: TypeAlias = Literal["ok", "skipped", "filtered", "needs_review", "failed"]

_DOCUMENT_TYPES = {"article", "filing", "earnings_call", "other"}
_TEXT_ROLES = {"title", "body", "section", "other"}
_STATUS_VALUES = {"ok", "skipped", "filtered", "needs_review", "failed"}


def _ensure_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"{context} must be an object/mapping, got {type(value)!r}")
    return value


def _reject_extra_keys(data: Mapping[str, Any], *, allowed: set[str], context: str) -> None:
    extra = sorted(set(data.keys()) - allowed)
    if extra:
        raise SchemaValidationError(f"{context} has unexpected keys: {extra}")


def _require_key(data: Mapping[str, Any], key: str, *, context: str) -> Any:
    if key not in data:
        raise SchemaValidationError(f"{context} is missing required key: {key}")
    return data[key]


def _require_str(data: Mapping[str, Any], key: str, *, context: str) -> str:
    value = _require_key(data, key, context=context)
    if not isinstance(value, str):
        raise SchemaValidationError(f"{context}.{key} must be a string")
    if value.strip() == "":
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


def _require_int(data: Mapping[str, Any], key: str, *, context: str) -> int:
    value = _require_key(data, key, context=context)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaValidationError(f"{context}.{key} must be an int")
    return value


def _optional_float(data: Mapping[str, Any], key: str, *, context: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{context}.{key} must be a number or null")
    return float(value)


def _optional_bool(data: Mapping[str, Any], key: str, *, context: str) -> bool | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise SchemaValidationError(f"{context}.{key} must be a bool or null")
    return value


def _require_float(data: Mapping[str, Any], key: str, *, context: str) -> float:
    value = _require_key(data, key, context=context)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{context}.{key} must be a number")
    return float(value)


def _optional_status(data: Mapping[str, Any], key: str, *, context: str) -> StatusValue | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SchemaValidationError(f"{context}.{key} must be a string or null")
    if value not in _STATUS_VALUES:
        raise SchemaValidationError(f"{context}.{key} must be one of {_STATUS_VALUES}")
    return value  # type: ignore[return-value]


def _require_status(data: Mapping[str, Any], key: str, *, context: str) -> StatusValue:
    value = _require_str(data, key, context=context)
    if value not in _STATUS_VALUES:
        raise SchemaValidationError(f"{context}.{key} must be one of {_STATUS_VALUES}")
    return value  # type: ignore[return-value]


def _require_text_role(data: Mapping[str, Any], key: str, *, context: str) -> TextRole:
    value = _require_str(data, key, context=context)
    if value not in _TEXT_ROLES:
        raise SchemaValidationError(f"{context}.{key} must be one of {_TEXT_ROLES}")
    return value  # type: ignore[return-value]


def _optional_document_type(
    data: Mapping[str, Any], key: str, *, context: str
) -> DocumentType | None:
    value = _optional_str(data, key, context=context)
    if value is None:
        return None
    if value not in _DOCUMENT_TYPES:
        raise SchemaValidationError(f"{context}.{key} must be one of {_DOCUMENT_TYPES} or null")
    return value  # type: ignore[return-value]


@dataclass(frozen=True)
class RowMetadata:
    title: str | None
    document_type: DocumentType | None
    text_role: TextRole
    original_id: str | None = None
    parent_id: str | None = None
    chunk_idx: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RowMetadata":
        data = _ensure_mapping(data, context="metadata")
        _reject_extra_keys(
            data,
            allowed={
                "title",
                "document_type",
                "text_role",
                "original_id",
                "parent_id",
                "chunk_idx",
            },
            context="metadata",
        )
        return cls(
            title=_optional_str(data, "title", context="metadata"),
            document_type=_optional_document_type(data, "document_type", context="metadata"),
            text_role=_require_text_role(data, "text_role", context="metadata"),
            original_id=_optional_str(data, "original_id", context="metadata"),
            parent_id=_optional_str(data, "parent_id", context="metadata"),
            chunk_idx=_optional_int(data, "chunk_idx", context="metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "document_type": self.document_type,
            "text_role": self.text_role,
            "original_id": self.original_id,
            "parent_id": self.parent_id,
            "chunk_idx": self.chunk_idx,
        }


@dataclass(frozen=True)
class NormalizedDatapoolRow:
    id: str
    dataset: str
    source: str
    metadata: RowMetadata

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NormalizedDatapoolRow":
        data = _ensure_mapping(data, context="normalized")
        _reject_extra_keys(
            data,
            allowed={"id", "dataset", "source", "metadata"},
            context="normalized",
        )
        metadata_value = _require_key(data, "metadata", context="normalized")
        return cls(
            id=_require_str(data, "id", context="normalized"),
            dataset=_require_str(data, "dataset", context="normalized"),
            source=_require_str(data, "source", context="normalized"),
            metadata=RowMetadata.from_dict(_ensure_mapping(metadata_value, context="metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "source": self.source,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True)
class Q1Row:
    id: str
    dataset: str
    source: str
    metadata: RowMetadata
    mt_q1: str
    qe_q1: float | None = None
    qe_raw_q1: float | None = None
    metricx_q1_clamped: bool | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Q1Row":
        data = _ensure_mapping(data, context="q1")
        _reject_extra_keys(
            data,
            allowed={
                "id",
                "dataset",
                "source",
                "metadata",
                "mt_q1",
                "qe_q1",
                "qe_raw_q1",
                "metricx_q1_clamped",
            },
            context="q1",
        )
        return cls(
            id=_require_str(data, "id", context="q1"),
            dataset=_require_str(data, "dataset", context="q1"),
            source=_require_str(data, "source", context="q1"),
            metadata=RowMetadata.from_dict(_ensure_mapping(_require_key(data, "metadata", context="q1"), context="metadata")),
            mt_q1=_require_str(data, "mt_q1", context="q1"),
            qe_q1=_optional_float(data, "qe_q1", context="q1"),
            qe_raw_q1=_optional_float(data, "qe_raw_q1", context="q1"),
            metricx_q1_clamped=_optional_bool(data, "metricx_q1_clamped", context="q1"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "source": self.source,
            "metadata": self.metadata.to_dict(),
            "mt_q1": self.mt_q1,
            "qe_q1": self.qe_q1,
            "qe_raw_q1": self.qe_raw_q1,
            "metricx_q1_clamped": self.metricx_q1_clamped,
        }


@dataclass(frozen=True)
class Q2Row:
    id: str
    dataset: str
    source: str
    metadata: RowMetadata
    mt_q2: str
    mt_q1: str | None = None
    qe_q1: float | None = None
    qe_raw_q1: float | None = None
    metricx_q1_clamped: bool | None = None
    qe_q2: float | None = None
    qe_raw_q2: float | None = None
    metricx_q2_clamped: bool | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Q2Row":
        data = _ensure_mapping(data, context="q2")
        _reject_extra_keys(
            data,
            allowed={
                "id",
                "dataset",
                "source",
                "metadata",
                "mt_q1",
                "mt_q2",
                "qe_q1",
                "qe_raw_q1",
                "metricx_q1_clamped",
                "qe_q2",
                "qe_raw_q2",
                "metricx_q2_clamped",
            },
            context="q2",
        )
        return cls(
            id=_require_str(data, "id", context="q2"),
            dataset=_require_str(data, "dataset", context="q2"),
            source=_require_str(data, "source", context="q2"),
            metadata=RowMetadata.from_dict(_ensure_mapping(_require_key(data, "metadata", context="q2"), context="metadata")),
            mt_q2=_require_str(data, "mt_q2", context="q2"),
            mt_q1=_optional_str(data, "mt_q1", context="q2"),
            qe_q1=_optional_float(data, "qe_q1", context="q2"),
            qe_raw_q1=_optional_float(data, "qe_raw_q1", context="q2"),
            metricx_q1_clamped=_optional_bool(data, "metricx_q1_clamped", context="q2"),
            qe_q2=_optional_float(data, "qe_q2", context="q2"),
            qe_raw_q2=_optional_float(data, "qe_raw_q2", context="q2"),
            metricx_q2_clamped=_optional_bool(data, "metricx_q2_clamped", context="q2"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "source": self.source,
            "metadata": self.metadata.to_dict(),
            "mt_q1": self.mt_q1,
            "mt_q2": self.mt_q2,
            "qe_q1": self.qe_q1,
            "qe_raw_q1": self.qe_raw_q1,
            "metricx_q1_clamped": self.metricx_q1_clamped,
            "qe_q2": self.qe_q2,
            "qe_raw_q2": self.qe_raw_q2,
            "metricx_q2_clamped": self.metricx_q2_clamped,
        }


@dataclass(frozen=True)
class ScoredRow:
    id: str
    dataset: str
    source: str
    metadata: RowMetadata
    score_s: float
    mt_q1: str | None = None
    mt_q2: str | None = None
    qe_q1: float | None = None
    qe_raw_q1: float | None = None
    metricx_q1_clamped: bool | None = None
    qe_q2: float | None = None
    qe_raw_q2: float | None = None
    metricx_q2_clamped: bool | None = None
    delta_qe: float | None = None
    collapse_term: float | None = None
    collapse_term_type: str | None = None
    difficulty_term: float | None = None
    collapse_z: float | None = None
    difficulty_z: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScoredRow":
        data = _ensure_mapping(data, context="scored")
        _reject_extra_keys(
            data,
            allowed={
                "id",
                "dataset",
                "source",
                "metadata",
                "mt_q1",
                "mt_q2",
                "qe_q1",
                "qe_raw_q1",
                "metricx_q1_clamped",
                "qe_q2",
                "qe_raw_q2",
                "metricx_q2_clamped",
                "delta_qe",
                "collapse_term",
                "collapse_term_type",
                "difficulty_term",
                "collapse_z",
                "difficulty_z",
                "score_s",
            },
            context="scored",
        )
        return cls(
            id=_require_str(data, "id", context="scored"),
            dataset=_require_str(data, "dataset", context="scored"),
            source=_require_str(data, "source", context="scored"),
            metadata=RowMetadata.from_dict(_ensure_mapping(_require_key(data, "metadata", context="scored"), context="metadata")),
            score_s=_require_float(data, "score_s", context="scored"),
            mt_q1=_optional_str(data, "mt_q1", context="scored"),
            mt_q2=_optional_str(data, "mt_q2", context="scored"),
            qe_q1=_optional_float(data, "qe_q1", context="scored"),
            qe_raw_q1=_optional_float(data, "qe_raw_q1", context="scored"),
            metricx_q1_clamped=_optional_bool(data, "metricx_q1_clamped", context="scored"),
            qe_q2=_optional_float(data, "qe_q2", context="scored"),
            qe_raw_q2=_optional_float(data, "qe_raw_q2", context="scored"),
            metricx_q2_clamped=_optional_bool(data, "metricx_q2_clamped", context="scored"),
            delta_qe=_optional_float(data, "delta_qe", context="scored"),
            collapse_term=_optional_float(data, "collapse_term", context="scored"),
            collapse_term_type=_optional_str(data, "collapse_term_type", context="scored"),
            difficulty_term=_optional_float(data, "difficulty_term", context="scored"),
            collapse_z=_optional_float(data, "collapse_z", context="scored"),
            difficulty_z=_optional_float(data, "difficulty_z", context="scored"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "source": self.source,
            "metadata": self.metadata.to_dict(),
            "score_s": self.score_s,
            "mt_q1": self.mt_q1,
            "mt_q2": self.mt_q2,
            "qe_q1": self.qe_q1,
            "qe_raw_q1": self.qe_raw_q1,
            "metricx_q1_clamped": self.metricx_q1_clamped,
            "qe_q2": self.qe_q2,
            "qe_raw_q2": self.qe_raw_q2,
            "metricx_q2_clamped": self.metricx_q2_clamped,
            "delta_qe": self.delta_qe,
            "collapse_term": self.collapse_term,
            "collapse_term_type": self.collapse_term_type,
            "difficulty_term": self.difficulty_term,
            "collapse_z": self.collapse_z,
            "difficulty_z": self.difficulty_z,
        }


@dataclass(frozen=True)
class SelectedRow:
    id: str
    dataset: str
    source: str
    metadata: RowMetadata
    score_s: float
    selection_rank: int
    selection_rule: str | None = None
    mt_q1: str | None = None
    mt_q2: str | None = None
    qe_q1: float | None = None
    qe_raw_q1: float | None = None
    metricx_q1_clamped: bool | None = None
    qe_q2: float | None = None
    qe_raw_q2: float | None = None
    metricx_q2_clamped: bool | None = None
    delta_qe: float | None = None
    collapse_term: float | None = None
    collapse_term_type: str | None = None
    difficulty_term: float | None = None
    collapse_z: float | None = None
    difficulty_z: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SelectedRow":
        data = _ensure_mapping(data, context="selected")
        _reject_extra_keys(
            data,
            allowed={
                "id",
                "dataset",
                "source",
                "metadata",
                "score_s",
                "selection_rank",
                "selection_rule",
                "mt_q1",
                "mt_q2",
                "qe_q1",
                "qe_raw_q1",
                "metricx_q1_clamped",
                "qe_q2",
                "qe_raw_q2",
                "metricx_q2_clamped",
                "delta_qe",
                "collapse_term",
                "collapse_term_type",
                "difficulty_term",
                "collapse_z",
                "difficulty_z",
            },
            context="selected",
        )
        selection_rank = _require_key(data, "selection_rank", context="selected")
        if isinstance(selection_rank, bool) or not isinstance(selection_rank, int):
            raise SchemaValidationError("selected.selection_rank must be an integer")
        return cls(
            id=_require_str(data, "id", context="selected"),
            dataset=_require_str(data, "dataset", context="selected"),
            source=_require_str(data, "source", context="selected"),
            metadata=RowMetadata.from_dict(_ensure_mapping(_require_key(data, "metadata", context="selected"), context="metadata")),
            score_s=_require_float(data, "score_s", context="selected"),
            selection_rank=selection_rank,
            selection_rule=_optional_str(data, "selection_rule", context="selected"),
            mt_q1=_optional_str(data, "mt_q1", context="selected"),
            mt_q2=_optional_str(data, "mt_q2", context="selected"),
            qe_q1=_optional_float(data, "qe_q1", context="selected"),
            qe_raw_q1=_optional_float(data, "qe_raw_q1", context="selected"),
            metricx_q1_clamped=_optional_bool(data, "metricx_q1_clamped", context="selected"),
            qe_q2=_optional_float(data, "qe_q2", context="selected"),
            qe_raw_q2=_optional_float(data, "qe_raw_q2", context="selected"),
            metricx_q2_clamped=_optional_bool(data, "metricx_q2_clamped", context="selected"),
            delta_qe=_optional_float(data, "delta_qe", context="selected"),
            collapse_term=_optional_float(data, "collapse_term", context="selected"),
            collapse_term_type=_optional_str(data, "collapse_term_type", context="selected"),
            difficulty_term=_optional_float(data, "difficulty_term", context="selected"),
            collapse_z=_optional_float(data, "collapse_z", context="selected"),
            difficulty_z=_optional_float(data, "difficulty_z", context="selected"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "source": self.source,
            "metadata": self.metadata.to_dict(),
            "score_s": self.score_s,
            "selection_rank": self.selection_rank,
            "selection_rule": self.selection_rule,
            "mt_q1": self.mt_q1,
            "mt_q2": self.mt_q2,
            "qe_q1": self.qe_q1,
            "qe_raw_q1": self.qe_raw_q1,
            "metricx_q1_clamped": self.metricx_q1_clamped,
            "qe_q2": self.qe_q2,
            "qe_raw_q2": self.qe_raw_q2,
            "metricx_q2_clamped": self.metricx_q2_clamped,
            "delta_qe": self.delta_qe,
            "collapse_term": self.collapse_term,
            "collapse_term_type": self.collapse_term_type,
            "difficulty_term": self.difficulty_term,
            "collapse_z": self.collapse_z,
            "difficulty_z": self.difficulty_z,
        }


@dataclass(frozen=True)
class ApiSelection:
    score_s: float
    qe_q1: float
    qe_q2: float
    delta_qe: float | None = None
    collapse_term: float | None = None
    collapse_term_type: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ApiSelection":
        data = _ensure_mapping(data, context="api_selection")
        _reject_extra_keys(
            data,
            allowed={"score_s", "qe_q1", "qe_q2", "delta_qe", "collapse_term", "collapse_term_type"},
            context="api_selection",
        )
        return cls(
            score_s=_require_float(data, "score_s", context="api_selection"),
            qe_q1=_require_float(data, "qe_q1", context="api_selection"),
            qe_q2=_require_float(data, "qe_q2", context="api_selection"),
            delta_qe=_optional_float(data, "delta_qe", context="api_selection"),
            collapse_term=_optional_float(data, "collapse_term", context="api_selection"),
            collapse_term_type=_optional_str(data, "collapse_term_type", context="api_selection"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_s": self.score_s,
            "qe_q1": self.qe_q1,
            "qe_q2": self.qe_q2,
            "delta_qe": self.delta_qe,
            "collapse_term": self.collapse_term,
            "collapse_term_type": self.collapse_term_type,
        }


@dataclass(frozen=True)
class ApiRequestRow:
    id: str
    row_id: str
    dataset: str
    source: str
    metadata: RowMetadata
    request_id: str
    run_id: str
    subset_idx: int
    student: str
    selection: ApiSelection
    prompt_version: str
    prompt_hash: str
    provider: str
    model: str
    status: StatusValue
    config_hash: str
    split_name: str | None = None
    # Free-form per-provider settings such as ``reasoning_effort`` /
    # ``thinking_mode``. May be omitted (``{}``) for legacy single-provider runs.
    model_params: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ApiRequestRow":
        data = _ensure_mapping(data, context="api_requests")
        _reject_extra_keys(
            data,
            allowed={
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
                "split_name",
                "model_params",
            },
            context="api_requests",
        )
        model_params_raw = data.get("model_params")
        if model_params_raw is None:
            model_params: dict[str, Any] | None = None
        elif isinstance(model_params_raw, Mapping):
            model_params = dict(model_params_raw)
        else:
            raise SchemaValidationError(
                "api_requests.model_params must be a mapping when present"
            )
        return cls(
            id=_require_str(data, "id", context="api_requests"),
            row_id=_require_str(data, "row_id", context="api_requests"),
            dataset=_require_str(data, "dataset", context="api_requests"),
            source=_require_str(data, "source", context="api_requests"),
            metadata=RowMetadata.from_dict(_ensure_mapping(_require_key(data, "metadata", context="api_requests"), context="metadata")),
            request_id=_require_str(data, "request_id", context="api_requests"),
            run_id=_require_str(data, "run_id", context="api_requests"),
            subset_idx=_require_int(data, "subset_idx", context="api_requests"),
            student=_require_str(data, "student", context="api_requests"),
            selection=ApiSelection.from_dict(
                _ensure_mapping(_require_key(data, "selection", context="api_requests"), context="api_selection")
            ),
            prompt_version=_require_str(data, "prompt_version", context="api_requests"),
            prompt_hash=_require_str(data, "prompt_hash", context="api_requests"),
            provider=_require_str(data, "provider", context="api_requests"),
            model=_require_str(data, "model", context="api_requests"),
            status=_require_status(data, "status", context="api_requests"),
            config_hash=_require_str(data, "config_hash", context="api_requests"),
            split_name=_optional_str(data, "split_name", context="api_requests"),
            model_params=model_params,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "row_id": self.row_id,
            "dataset": self.dataset,
            "source": self.source,
            "metadata": self.metadata.to_dict(),
            "request_id": self.request_id,
            "run_id": self.run_id,
            "subset_idx": self.subset_idx,
            "student": self.student,
            "selection": self.selection.to_dict(),
            "prompt_version": self.prompt_version,
            "prompt_hash": self.prompt_hash,
            "provider": self.provider,
            "model": self.model,
            "split_name": self.split_name,
            "model_params": self.model_params,
            "status": self.status,
            "config_hash": self.config_hash,
        }


@dataclass(frozen=True)
class ApiUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    # Number of reasoning / thinking tokens included inside ``output_tokens``.
    # Optional; defaults to 0 when the provider does not expose a count or when
    # the request did not trigger reasoning.
    reasoning_tokens: int = 0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ApiUsage":
        data = _ensure_mapping(data, context="api_usage")
        _reject_extra_keys(
            data,
            allowed={"input_tokens", "output_tokens", "total_tokens", "reasoning_tokens"},
            context="api_usage",
        )
        reasoning_value = data.get("reasoning_tokens", 0)
        if reasoning_value is None:
            reasoning_tokens = 0
        else:
            try:
                reasoning_tokens = int(reasoning_value)
            except (TypeError, ValueError) as exc:
                raise SchemaValidationError(
                    "api_usage.reasoning_tokens must be an integer"
                ) from exc
        if reasoning_tokens < 0:
            raise SchemaValidationError(
                "api_usage.reasoning_tokens must be >= 0"
            )
        return cls(
            input_tokens=_require_int(data, "input_tokens", context="api_usage"),
            output_tokens=_require_int(data, "output_tokens", context="api_usage"),
            total_tokens=_require_int(data, "total_tokens", context="api_usage"),
            reasoning_tokens=reasoning_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


@dataclass(frozen=True)
class ApiCost:
    currency: str
    estimated: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ApiCost":
        data = _ensure_mapping(data, context="api_cost")
        _reject_extra_keys(data, allowed={"currency", "estimated"}, context="api_cost")
        return cls(
            currency=_require_str(data, "currency", context="api_cost"),
            estimated=_require_float(data, "estimated", context="api_cost"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "estimated": self.estimated,
        }


@dataclass(frozen=True)
class ApiRow:
    id: str
    row_id: str
    dataset: str
    source: str
    metadata: RowMetadata
    request_id: str
    run_id: str
    subset_idx: int
    provider: str
    model: str
    status: StatusValue
    teacher_label: str
    student: str
    gold: str | None = None
    reason: str | None = None
    prompt_version: str | None = None
    prompt_hash: str | None = None
    usage: ApiUsage | None = None
    cost: ApiCost | None = None
    latency_ms: float | None = None
    attempt: int | None = None
    error: str | None = None
    config_hash: str | None = None
    # Routing metadata (optional; populated when external_api.routing.mode='weighted').
    split_name: str | None = None
    # Summarized chain-of-thought returned by Claude adaptive / Gemini dynamic
    # thinking; empty for providers that hide reasoning (OpenAI) or for off mode.
    thinking_text: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ApiRow":
        data = _ensure_mapping(data, context="api")
        _reject_extra_keys(
            data,
            allowed={
                "id",
                "row_id",
                "dataset",
                "source",
                "metadata",
                "request_id",
                "run_id",
                "subset_idx",
                "provider",
                "model",
                "status",
                "teacher_label",
                "student",
                "gold",
                "reason",
                "prompt_version",
                "prompt_hash",
                "usage",
                "cost",
                "latency_ms",
                "attempt",
                "error",
                "config_hash",
                "split_name",
                "thinking_text",
            },
            context="api",
        )
        status = _require_status(data, "status", context="api")
        gold = _optional_str(data, "gold", context="api")
        if status == "ok" and (gold is None or not gold.strip()):
            raise SchemaValidationError("api.gold is required when api.status=ok")
        reason = _optional_str(data, "reason", context="api")
        if status != "ok" and (reason is None or not reason.strip()):
            raise SchemaValidationError("api.reason is required when api.status!=ok")

        usage_data = data.get("usage")
        usage = None
        if usage_data is not None:
            usage = ApiUsage.from_dict(_ensure_mapping(usage_data, context="api_usage"))
        cost_data = data.get("cost")
        cost = None
        if cost_data is not None:
            cost = ApiCost.from_dict(_ensure_mapping(cost_data, context="api_cost"))
        return cls(
            id=_require_str(data, "id", context="api"),
            row_id=_require_str(data, "row_id", context="api"),
            dataset=_require_str(data, "dataset", context="api"),
            source=_require_str(data, "source", context="api"),
            metadata=RowMetadata.from_dict(_ensure_mapping(_require_key(data, "metadata", context="api"), context="metadata")),
            request_id=_require_str(data, "request_id", context="api"),
            run_id=_require_str(data, "run_id", context="api"),
            subset_idx=_require_int(data, "subset_idx", context="api"),
            provider=_require_str(data, "provider", context="api"),
            model=_require_str(data, "model", context="api"),
            status=status,
            teacher_label=_require_str(data, "teacher_label", context="api"),
            student=_require_str(data, "student", context="api"),
            gold=gold,
            reason=reason,
            prompt_version=_optional_str(data, "prompt_version", context="api"),
            prompt_hash=_optional_str(data, "prompt_hash", context="api"),
            usage=usage,
            cost=cost,
            latency_ms=_optional_float(data, "latency_ms", context="api"),
            attempt=_optional_int(data, "attempt", context="api"),
            error=_optional_str(data, "error", context="api"),
            config_hash=_optional_str(data, "config_hash", context="api"),
            split_name=_optional_str(data, "split_name", context="api"),
            thinking_text=_optional_str(data, "thinking_text", context="api"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "row_id": self.row_id,
            "dataset": self.dataset,
            "source": self.source,
            "metadata": self.metadata.to_dict(),
            "request_id": self.request_id,
            "run_id": self.run_id,
            "subset_idx": self.subset_idx,
            "provider": self.provider,
            "model": self.model,
            "teacher_label": self.teacher_label,
            "student": self.student,
            "gold": self.gold,
            "status": self.status,
            "reason": self.reason,
            "split_name": self.split_name,
            "thinking_text": self.thinking_text,
            "prompt_version": self.prompt_version,
            "prompt_hash": self.prompt_hash,
            "usage": self.usage.to_dict() if self.usage is not None else None,
            "cost": self.cost.to_dict() if self.cost is not None else None,
            "latency_ms": self.latency_ms,
            "attempt": self.attempt,
            "error": self.error,
            "config_hash": self.config_hash,
        }


@dataclass(frozen=True)
class PreferencePairRow:
    id: str
    row_id: str
    request_id: str
    run_id: str
    subset_idx: int
    dataset: str
    source: str
    metadata: RowMetadata
    student: str
    gold: str | None
    status: StatusValue
    error_type: str
    teacher_label: str
    reason: str | None = None
    error: str | None = None
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    prompt_hash: str | None = None
    usage: ApiUsage | None = None
    cost: ApiCost | None = None
    latency_ms: float | None = None
    attempt: int | None = None
    config_hash: str | None = None
    split_name: str | None = None
    thinking_text: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PreferencePairRow":
        data = _ensure_mapping(data, context="preference_pairs")
        _reject_extra_keys(
            data,
            allowed={
                "id",
                "row_id",
                "request_id",
                "run_id",
                "subset_idx",
                "dataset",
                "source",
                "metadata",
                "student",
                "gold",
                "status",
                "error_type",
                "teacher_label",
                "reason",
                "error",
                "provider",
                "model",
                "prompt_version",
                "prompt_hash",
                "usage",
                "cost",
                "latency_ms",
                "attempt",
                "config_hash",
                "split_name",
                "thinking_text",
            },
            context="preference_pairs",
        )
        status = _require_status(data, "status", context="preference_pairs")
        gold = _optional_str(data, "gold", context="preference_pairs")
        if status == "ok" and (gold is None or not gold.strip()):
            raise SchemaValidationError(
                "preference_pairs.gold is required when preference_pairs.status=ok"
            )

        error_type = _require_str(data, "error_type", context="preference_pairs")
        if status == "ok" and error_type != "none":
            raise SchemaValidationError(
                "preference_pairs.error_type must be 'none' when preference_pairs.status=ok"
            )
        if status != "ok" and error_type == "none":
            raise SchemaValidationError(
                "preference_pairs.error_type must not be 'none' when preference_pairs.status!=ok"
            )

        usage_data = data.get("usage")
        usage = None
        if usage_data is not None:
            usage = ApiUsage.from_dict(_ensure_mapping(usage_data, context="api_usage"))
        cost_data = data.get("cost")
        cost = None
        if cost_data is not None:
            cost = ApiCost.from_dict(_ensure_mapping(cost_data, context="api_cost"))
        return cls(
            id=_require_str(data, "id", context="preference_pairs"),
            row_id=_require_str(data, "row_id", context="preference_pairs"),
            request_id=_require_str(data, "request_id", context="preference_pairs"),
            run_id=_require_str(data, "run_id", context="preference_pairs"),
            subset_idx=_require_int(data, "subset_idx", context="preference_pairs"),
            dataset=_require_str(data, "dataset", context="preference_pairs"),
            source=_require_str(data, "source", context="preference_pairs"),
            metadata=RowMetadata.from_dict(
                _ensure_mapping(
                    _require_key(data, "metadata", context="preference_pairs"),
                    context="metadata",
                )
            ),
            student=_require_str(data, "student", context="preference_pairs"),
            gold=gold,
            status=status,
            error_type=error_type,
            teacher_label=_require_str(data, "teacher_label", context="preference_pairs"),
            reason=_optional_str(data, "reason", context="preference_pairs"),
            error=_optional_str(data, "error", context="preference_pairs"),
            provider=_optional_str(data, "provider", context="preference_pairs"),
            model=_optional_str(data, "model", context="preference_pairs"),
            prompt_version=_optional_str(data, "prompt_version", context="preference_pairs"),
            prompt_hash=_optional_str(data, "prompt_hash", context="preference_pairs"),
            usage=usage,
            cost=cost,
            latency_ms=_optional_float(data, "latency_ms", context="preference_pairs"),
            attempt=_optional_int(data, "attempt", context="preference_pairs"),
            config_hash=_optional_str(data, "config_hash", context="preference_pairs"),
            split_name=_optional_str(data, "split_name", context="preference_pairs"),
            thinking_text=_optional_str(data, "thinking_text", context="preference_pairs"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "row_id": self.row_id,
            "request_id": self.request_id,
            "run_id": self.run_id,
            "subset_idx": self.subset_idx,
            "dataset": self.dataset,
            "source": self.source,
            "metadata": self.metadata.to_dict(),
            "student": self.student,
            "gold": self.gold,
            "status": self.status,
            "error_type": self.error_type,
            "teacher_label": self.teacher_label,
            "reason": self.reason,
            "error": self.error,
            "provider": self.provider,
            "model": self.model,
            "split_name": self.split_name,
            "thinking_text": self.thinking_text,
            "prompt_version": self.prompt_version,
            "prompt_hash": self.prompt_hash,
            "usage": self.usage.to_dict() if self.usage is not None else None,
            "cost": self.cost.to_dict() if self.cost is not None else None,
            "latency_ms": self.latency_ms,
            "attempt": self.attempt,
            "config_hash": self.config_hash,
        }


@dataclass(frozen=True)
class TrainRow:
    id: str
    dataset: str
    source: str
    gold: str
    metadata: RowMetadata

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainRow":
        data = _ensure_mapping(data, context="train")
        _reject_extra_keys(
            data,
            allowed={"id", "dataset", "source", "gold", "metadata"},
            context="train",
        )
        return cls(
            id=_require_str(data, "id", context="train"),
            dataset=_require_str(data, "dataset", context="train"),
            source=_require_str(data, "source", context="train"),
            gold=_require_str(data, "gold", context="train"),
            metadata=RowMetadata.from_dict(_ensure_mapping(_require_key(data, "metadata", context="train"), context="metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "source": self.source,
            "gold": self.gold,
            "metadata": self.metadata.to_dict(),
        }


_ARTIFACT_TO_MODEL = {
    "normalized": NormalizedDatapoolRow,
    "input": NormalizedDatapoolRow,
    "q1": Q1Row,
    "q2": Q2Row,
    "scored": ScoredRow,
    "selected": SelectedRow,
    "api_requests": ApiRequestRow,
    "api": ApiRow,
    "preference_pairs": PreferencePairRow,
    "train": TrainRow,
}


def validate_artifact_row(row: Mapping[str, Any], artifact: ArtifactName) -> dict[str, Any]:
    model_cls = _ARTIFACT_TO_MODEL[artifact]
    model = model_cls.from_dict(row)
    return model.to_dict()


def validate_artifact_rows(
    rows: Iterable[Mapping[str, Any]], artifact: ArtifactName
) -> list[dict[str, Any]]:
    return [validate_artifact_row(row, artifact) for row in rows]
