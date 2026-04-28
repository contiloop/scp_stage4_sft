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


def _optional_float(data: Mapping[str, Any], key: str, *, context: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{context}.{key} must be a number or null")
    return float(value)


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

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Q1Row":
        data = _ensure_mapping(data, context="q1")
        _reject_extra_keys(
            data,
            allowed={"id", "dataset", "source", "metadata", "mt_q1", "qe_q1"},
            context="q1",
        )
        return cls(
            id=_require_str(data, "id", context="q1"),
            dataset=_require_str(data, "dataset", context="q1"),
            source=_require_str(data, "source", context="q1"),
            metadata=RowMetadata.from_dict(_ensure_mapping(_require_key(data, "metadata", context="q1"), context="metadata")),
            mt_q1=_require_str(data, "mt_q1", context="q1"),
            qe_q1=_optional_float(data, "qe_q1", context="q1"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "source": self.source,
            "metadata": self.metadata.to_dict(),
            "mt_q1": self.mt_q1,
            "qe_q1": self.qe_q1,
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
    qe_q2: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Q2Row":
        data = _ensure_mapping(data, context="q2")
        _reject_extra_keys(
            data,
            allowed={"id", "dataset", "source", "metadata", "mt_q1", "mt_q2", "qe_q1", "qe_q2"},
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
            qe_q2=_optional_float(data, "qe_q2", context="q2"),
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
            "qe_q2": self.qe_q2,
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
    qe_q2: float | None = None

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
                "qe_q2",
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
            qe_q2=_optional_float(data, "qe_q2", context="scored"),
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
            "qe_q2": self.qe_q2,
        }


@dataclass(frozen=True)
class SelectedRow:
    id: str
    dataset: str
    source: str
    metadata: RowMetadata
    score_s: float
    selection_rank: int
    mt_q1: str | None = None
    mt_q2: str | None = None
    qe_q1: float | None = None
    qe_q2: float | None = None

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
                "mt_q1",
                "mt_q2",
                "qe_q1",
                "qe_q2",
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
            mt_q1=_optional_str(data, "mt_q1", context="selected"),
            mt_q2=_optional_str(data, "mt_q2", context="selected"),
            qe_q1=_optional_float(data, "qe_q1", context="selected"),
            qe_q2=_optional_float(data, "qe_q2", context="selected"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "source": self.source,
            "metadata": self.metadata.to_dict(),
            "score_s": self.score_s,
            "selection_rank": self.selection_rank,
            "mt_q1": self.mt_q1,
            "mt_q2": self.mt_q2,
            "qe_q1": self.qe_q1,
            "qe_q2": self.qe_q2,
        }


@dataclass(frozen=True)
class ApiRequestRow:
    id: str
    dataset: str
    source: str
    metadata: RowMetadata
    request_id: str
    student: str
    status: StatusValue

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ApiRequestRow":
        data = _ensure_mapping(data, context="api_requests")
        _reject_extra_keys(
            data,
            allowed={"id", "dataset", "source", "metadata", "request_id", "student", "status"},
            context="api_requests",
        )
        return cls(
            id=_require_str(data, "id", context="api_requests"),
            dataset=_require_str(data, "dataset", context="api_requests"),
            source=_require_str(data, "source", context="api_requests"),
            metadata=RowMetadata.from_dict(_ensure_mapping(_require_key(data, "metadata", context="api_requests"), context="metadata")),
            request_id=_require_str(data, "request_id", context="api_requests"),
            student=_require_str(data, "student", context="api_requests"),
            status=_require_status(data, "status", context="api_requests"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "source": self.source,
            "metadata": self.metadata.to_dict(),
            "request_id": self.request_id,
            "student": self.student,
            "status": self.status,
        }


@dataclass(frozen=True)
class ApiRow:
    id: str
    dataset: str
    source: str
    metadata: RowMetadata
    request_id: str
    gold: str
    status: StatusValue

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ApiRow":
        data = _ensure_mapping(data, context="api")
        _reject_extra_keys(
            data,
            allowed={"id", "dataset", "source", "metadata", "request_id", "gold", "status"},
            context="api",
        )
        status = _require_status(data, "status", context="api")
        return cls(
            id=_require_str(data, "id", context="api"),
            dataset=_require_str(data, "dataset", context="api"),
            source=_require_str(data, "source", context="api"),
            metadata=RowMetadata.from_dict(_ensure_mapping(_require_key(data, "metadata", context="api"), context="metadata")),
            request_id=_require_str(data, "request_id", context="api"),
            gold=_require_str(data, "gold", context="api"),
            status=status,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "source": self.source,
            "metadata": self.metadata.to_dict(),
            "request_id": self.request_id,
            "gold": self.gold,
            "status": self.status,
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
