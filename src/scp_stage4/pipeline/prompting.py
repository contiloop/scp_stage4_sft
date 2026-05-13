"""Prompt config resolution and rendering helpers for Stage 4 workers."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

_DEFAULT_TRANSLATION_VERSION = "translation_v1"
_DEFAULT_TRANSLATION_TEMPLATES = [
    (
        "You are a professional {src_lang_name} ({src_locale}) to {tgt_lang_name} "
        "({tgt_locale}) translator. Your goal is to accurately convey the meaning and "
        "nuances of the original {src_lang_name} text while adhering to {tgt_lang_name} "
        "grammar, vocabulary, and cultural sensitivities. Produce only the "
        "{tgt_lang_name} translation, without any additional explanations or commentary. "
        "Please translate the following {src_lang_name} text into {tgt_lang_name}: {src}"
    ),
    "Translate the following text from {src_lang_name} to {tgt_lang_name}: {src}",
    "What does this sentence mean in {tgt_lang_name} from {src_lang_name}: {src}",
    "How do you translate this sentence into {tgt_lang_name} from {src_lang_name}: {src}",
    "Translate the following text to {tgt_lang_name}: {src}",
]
_DEFAULT_TRANSLATION_SELECTION_SCOPE = "row_id"
_DEFAULT_TRANSLATION_SEED = 42

_DEFAULT_SFT_VERSION = "sft_translation_v1"
_DEFAULT_SFT_INSTRUCTION_TEMPLATE = (
    "### Instruction:\n"
    "Translate the English source into Korean.\n\n"
    "### Source:\n"
    "{source}\n\n"
)
_DEFAULT_SFT_RESPONSE_TEMPLATE = "### Response:\n"

_DEFAULT_TEACHER_VERSION = "teacher_correction_v1"
_DEFAULT_TEACHER_ALLOWED_FIELDS = ("dataset", "document_type", "text_role", "title")


class PromptConfigError(ValueError):
    """Raised when prompt configuration or rendering is invalid."""


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _non_empty_str(value: Any) -> str | None:
    if isinstance(value, str):
        if value.strip():
            return value
    return None


def _normalized_non_empty_str(value: Any) -> str | None:
    text = _non_empty_str(value)
    if text is None:
        return None
    return text.strip()


def _as_non_empty_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = _non_empty_str(item)
        if text is not None:
            out.append(text)
    return out


def stable_template_index(sample_key: str, template_count: int, seed: int) -> int:
    """Stable row-level template assignment (stage3-compatible hash policy)."""
    if template_count <= 0:
        raise PromptConfigError("template_count must be positive")
    key = f"{seed}|{sample_key}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return value % template_count


def _prompts_cfg(prompts: Mapping[str, Any] | None) -> dict[str, Any]:
    return _as_dict(prompts)


def _translation_cfg(prompts: Mapping[str, Any] | None) -> dict[str, Any]:
    return _as_dict(_prompts_cfg(prompts).get("translation"))


def _sft_cfg(prompts: Mapping[str, Any] | None) -> dict[str, Any]:
    return _as_dict(_prompts_cfg(prompts).get("sft"))


def _teacher_cfg(prompts: Mapping[str, Any] | None) -> dict[str, Any]:
    root = _prompts_cfg(prompts)
    scoped = _as_dict(root.get("teacher_correction"))
    if not scoped:
        raise PromptConfigError("prompts.teacher_correction config is required")
    return scoped


def translation_prompt_version(prompts: Mapping[str, Any] | None) -> str:
    cfg = _translation_cfg(prompts)
    return _normalized_non_empty_str(cfg.get("version")) or _DEFAULT_TRANSLATION_VERSION


def teacher_prompt_version(prompts: Mapping[str, Any] | None) -> str:
    cfg = _teacher_cfg(prompts)
    return _normalized_non_empty_str(cfg.get("version")) or _DEFAULT_TEACHER_VERSION


def teacher_prompt_hash(prompts: Mapping[str, Any] | None) -> str:
    cfg = _teacher_cfg(prompts)
    canonical = json.dumps(cfg, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_translation_templates(prompts: Mapping[str, Any] | None) -> list[str]:
    cfg = _translation_cfg(prompts)
    templates = _as_non_empty_str_list(cfg.get("templates"))
    if templates:
        return templates
    return list(_DEFAULT_TRANSLATION_TEMPLATES)


def _resolve_template_seed(prompts: Mapping[str, Any] | None) -> int:
    cfg = _translation_cfg(prompts)
    raw_seed = cfg.get("template_seed", _DEFAULT_TRANSLATION_SEED)
    if isinstance(raw_seed, bool):
        return _DEFAULT_TRANSLATION_SEED
    if isinstance(raw_seed, int):
        return raw_seed
    if isinstance(raw_seed, str):
        text = raw_seed.strip()
        if text:
            try:
                return int(text)
            except ValueError:
                pass
    return _DEFAULT_TRANSLATION_SEED


def _resolve_selection_scope(prompts: Mapping[str, Any] | None) -> str:
    cfg = _translation_cfg(prompts)
    scope = _normalized_non_empty_str(cfg.get("selection_seed_scope"))
    if scope is not None:
        scope = scope.lower()
    if scope not in {"row_id", "row_id_subset"}:
        return _DEFAULT_TRANSLATION_SELECTION_SCOPE
    return scope


def _resolve_fixed_template_index(prompts: Mapping[str, Any] | None) -> int | None:
    cfg = _translation_cfg(prompts)
    raw = cfg.get("fixed_template_index")
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if text:
            try:
                return int(text)
            except ValueError:
                return None
    return None


def _translation_placeholders(
    prompts: Mapping[str, Any] | None,
    *,
    source: str,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    cfg = _translation_cfg(prompts)
    metadata_payload = _as_dict(metadata)
    metadata_json = json.dumps(metadata_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "src_lang_name": _normalized_non_empty_str(cfg.get("src_lang_name")) or "English",
        "tgt_lang_name": _normalized_non_empty_str(cfg.get("tgt_lang_name")) or "Korean",
        "src_locale": _normalized_non_empty_str(cfg.get("src_locale")) or "en-US",
        "tgt_locale": _normalized_non_empty_str(cfg.get("tgt_locale")) or "ko-KR",
        "src": source,
        "source": source,
        "metadata_json": metadata_json,
    }


def _render_with_template(template: str, values: Mapping[str, Any], *, context: str) -> str:
    try:
        return template.format(**values)
    except KeyError as exc:
        missing = str(exc).strip("'")
        raise PromptConfigError(
            f"{context} template requires missing placeholder: {missing}"
        ) from exc


def _is_non_empty_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def render_translation_prompt(
    *,
    prompts: Mapping[str, Any] | None,
    source: str,
    row_id: str,
    subset_idx: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[str, int]:
    templates = resolve_translation_templates(prompts)
    fixed_index = _resolve_fixed_template_index(prompts)
    if fixed_index is not None:
        if fixed_index < 0 or fixed_index >= len(templates):
            raise PromptConfigError(
                f"prompts.translation.fixed_template_index={fixed_index} is out of range"
            )
        template_index = fixed_index
    else:
        seed = _resolve_template_seed(prompts)
        scope = _resolve_selection_scope(prompts)
        if scope == "row_id_subset":
            sample_key = f"{row_id}:{subset_idx if subset_idx is not None else 0}"
        else:
            sample_key = row_id
        template_index = stable_template_index(
            sample_key=sample_key,
            template_count=len(templates),
            seed=seed,
        )
    template = templates[template_index]
    placeholders = _translation_placeholders(prompts, source=source, metadata=metadata)
    return _render_with_template(
        template,
        placeholders,
        context="translation",
    ), template_index


def sft_instruction_template(prompts: Mapping[str, Any] | None) -> str:
    cfg = _sft_cfg(prompts)
    return _non_empty_str(cfg.get("instruction_template")) or _DEFAULT_SFT_INSTRUCTION_TEMPLATE


def sft_response_template(prompts: Mapping[str, Any] | None) -> str:
    cfg = _sft_cfg(prompts)
    return _non_empty_str(cfg.get("response_template")) or _DEFAULT_SFT_RESPONSE_TEMPLATE


def sft_prompt_version(prompts: Mapping[str, Any] | None) -> str:
    cfg = _sft_cfg(prompts)
    return _non_empty_str(cfg.get("version")) or _DEFAULT_SFT_VERSION


def render_sft_text(
    *,
    prompts: Mapping[str, Any] | None,
    source: str,
    target: str,
) -> str:
    instruction_template = sft_instruction_template(prompts)
    response_template = sft_response_template(prompts)
    instruction = _render_with_template(
        instruction_template,
        {"source": source},
        context="sft instruction",
    )
    return f"{instruction}{response_template}{target}"


def _render_teacher_metadata(
    row: Mapping[str, Any],
    teacher_cfg: Mapping[str, Any],
) -> str:
    metadata_cfg = _as_dict(teacher_cfg.get("metadata"))
    include = metadata_cfg.get("include", True)
    if isinstance(include, bool) and not include:
        return "{}"

    metadata = _as_dict(row.get("metadata"))
    combined: dict[str, Any] = dict(metadata)
    dataset = row.get("dataset")
    if isinstance(dataset, str) and dataset.strip():
        combined["dataset"] = dataset

    allowed_fields = _as_non_empty_str_list(metadata_cfg.get("allowed_fields"))
    if allowed_fields:
        filtered = {
            key: combined[key]
            for key in allowed_fields
            if key in combined and _is_non_empty_value(combined[key])
        }
    else:
        filtered = {
            key: value
            for key, value in combined.items()
            if _is_non_empty_value(value)
        }
        if not filtered:
            filtered = {
                key: value
                for key, value in combined.items()
                if key in _DEFAULT_TEACHER_ALLOWED_FIELDS and _is_non_empty_value(value)
            }

    render_format = _normalized_non_empty_str(metadata_cfg.get("render_format")) or "json"
    if render_format == "json":
        return json.dumps(filtered, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if render_format == "kv":
        if not filtered:
            return "{}"
        return "\n".join(f"{key}: {value}" for key, value in sorted(filtered.items()))
    raise PromptConfigError(
        "prompts.teacher_correction.metadata.render_format must be one of: json, kv"
    )


def teacher_system_prompt(prompts: Mapping[str, Any] | None) -> str:
    cfg = _teacher_cfg(prompts)
    system_template = _non_empty_str(cfg.get("system_template"))
    if system_template is None:
        raise PromptConfigError("prompts.teacher_correction.system_template must be a non-empty string")
    return system_template


def render_teacher_user_prompt(
    *,
    prompts: Mapping[str, Any] | None,
    row: Mapping[str, Any],
) -> str:
    cfg = _teacher_cfg(prompts)
    template = _non_empty_str(cfg.get("user_template"))
    if template is None:
        raise PromptConfigError("prompts.teacher_correction.user_template must be a non-empty string")
    source = str(row.get("source", ""))
    student = str(row.get("student", ""))
    metadata_text = _render_teacher_metadata(row, cfg)
    return _render_with_template(
        template,
        {
            "source": source,
            "student": student,
            "metadata_json": metadata_text,
            "metadata": metadata_text,
            "dataset": str(row.get("dataset", "")),
        },
        context="teacher_correction",
    )
