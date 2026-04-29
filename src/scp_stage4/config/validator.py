"""Validation rules for composed SCP Stage 4 config."""

from __future__ import annotations

import re
from typing import Any


class ConfigValidationError(ValueError):
    """Raised when config violates required contracts."""


_REQUIRED_TOP_LEVEL = (
    "model",
    "data",
    "inference",
    "pipeline",
    "training",
    "qe",
    "external_api",
    "logging",
    "prompts",
    "run",
)

_REQUIRED_LOG_FIELDS = ("run_id", "subset_idx", "phase", "config_hash")
_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_OVERFLOW_POLICIES = {"split", "skip", "truncate"}
_SPLIT_UNIT_POLICIES = {"sentence"}
_SPLIT_LONG_SENTENCE_POLICIES = {"skip", "truncate", "split"}
_SPLIT_MAX_CHUNKS_EXCEEDED_POLICIES = {"skip", "error"}
_SUBSET_ARCHIVE_FORMATS = {"tar", "tar.gz", "tar.xz"}
_DATA_RUNTIME_MODES = {"fixture", "hf", "local_jsonl"}
_INFERENCE_RUNTIME_MODES = {"mock", "subprocess"}
_QE_RUNTIME_MODES = {"mock", "subprocess"}
_API_RUNTIME_MODES = {"mock", "subprocess"}
_TRAINING_RUNTIME_MODES = {"mock", "subprocess"}


def _err(errors: list[str], message: str) -> None:
    errors.append(message)


def _as_dict(value: Any, name: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        _err(errors, f"{name} must be a mapping")
        return {}
    return value


def _require_number(
    cfg: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    allow_zero: bool = False,
) -> float | None:
    value = cfg.get(key)
    if not isinstance(value, (int, float)):
        _err(errors, f"{key} must be numeric")
        return None
    if allow_zero:
        if value < 0:
            _err(errors, f"{key} must be >= 0")
            return None
    elif value <= 0:
        _err(errors, f"{key} must be > 0")
        return None
    return float(value)


def _validate_external_api_env_names(external_api: dict[str, Any], errors: list[str]) -> None:
    primary = _as_dict(external_api.get("primary", {}), "external_api.primary", errors)
    api_key_env = primary.get("api_key_env")
    if not isinstance(api_key_env, str) or not _ENV_NAME_RE.match(api_key_env):
        _err(errors, "external_api.primary.api_key_env must be an env var name")

    providers = _as_dict(
        external_api.get("providers", {}), "external_api.providers", errors
    )
    for provider_name, provider_cfg in providers.items():
        if not isinstance(provider_cfg, dict):
            _err(errors, f"external_api.providers.{provider_name} must be a mapping")
            continue
        value = provider_cfg.get("api_key_env")
        if not isinstance(value, str) or not _ENV_NAME_RE.match(value):
            _err(
                errors,
                f"external_api.providers.{provider_name}.api_key_env must be an env var name",
            )


def _validate_subprocess_command(
    section_name: str,
    section_cfg: dict[str, Any],
    mode_key: str,
    command_key: str,
    allowed_modes: set[str],
    errors: list[str],
) -> None:
    runtime = _as_dict(section_cfg.get("runtime", {}), f"{section_name}.runtime", errors)
    mode = runtime.get(mode_key)
    if not isinstance(mode, str) or mode not in allowed_modes:
        _err(
            errors,
            f"{section_name}.runtime.{mode_key} must be one of: {', '.join(sorted(allowed_modes))}",
        )
        return

    subprocess_cfg = _as_dict(
        runtime.get("subprocess", {}), f"{section_name}.runtime.subprocess", errors
    )
    command = subprocess_cfg.get(command_key)
    if mode != "subprocess":
        return
    if not isinstance(command, list) or not command:
        _err(
            errors,
            f"{section_name}.runtime.subprocess.{command_key} must be a non-empty list when mode=subprocess",
        )
        return
    for idx, part in enumerate(command):
        if not isinstance(part, str) or not part.strip():
            _err(
                errors,
                f"{section_name}.runtime.subprocess.{command_key}[{idx}] must be a non-empty string",
            )


def _validate_training_runtime(training: dict[str, Any], errors: list[str]) -> None:
    runtime = _as_dict(training.get("runtime", {}), "training.runtime", errors)
    mode = runtime.get("mode")
    if not isinstance(mode, str) or mode not in _TRAINING_RUNTIME_MODES:
        _err(
            errors,
            "training.runtime.mode must be one of: "
            + ", ".join(sorted(_TRAINING_RUNTIME_MODES)),
        )
        return

    subprocess_cfg = _as_dict(
        runtime.get("subprocess", {}), "training.runtime.subprocess", errors
    )
    if mode != "subprocess":
        return
    for key in ("collapse_command", "unload_command", "update_command"):
        command = subprocess_cfg.get(key)
        if not isinstance(command, list) or not command:
            _err(
                errors,
                f"training.runtime.subprocess.{key} must be a non-empty list when mode=subprocess",
            )
            continue
        for idx, part in enumerate(command):
            if not isinstance(part, str) or not part.strip():
                _err(
                    errors,
                    f"training.runtime.subprocess.{key}[{idx}] must be a non-empty string",
                )


def validate_config(cfg: dict[str, Any]) -> None:
    errors: list[str] = []

    for key in _REQUIRED_TOP_LEVEL:
        if key not in cfg:
            _err(errors, f"Missing required top-level section: {key}")

    model = _as_dict(cfg.get("model", {}), "model", errors)
    data = _as_dict(cfg.get("data", {}), "data", errors)
    inference = _as_dict(cfg.get("inference", {}), "inference", errors)
    pipeline = _as_dict(cfg.get("pipeline", {}), "pipeline", errors)
    training = _as_dict(cfg.get("training", {}), "training", errors)
    external_api = _as_dict(cfg.get("external_api", {}), "external_api", errors)
    logging_cfg = _as_dict(cfg.get("logging", {}), "logging", errors)

    max_length = _require_number(model, "max_length", errors)
    max_seq_length = model.get("max_seq_length")
    if max_seq_length is None and max_length is not None:
        max_seq_length = max_length
        model["max_seq_length"] = max_seq_length

    if max_seq_length is not None:
        if not isinstance(max_seq_length, (int, float)):
            _err(errors, "model.max_seq_length must be numeric or null")
        elif max_seq_length <= 0:
            _err(errors, "model.max_seq_length must be > 0")
        elif max_length is not None and max_seq_length > max_length:
            _err(errors, "model.max_seq_length must be <= model.max_length")

    length_cfg = _as_dict(data.get("length", {}), "data.length", errors)
    data_runtime = _as_dict(data.get("runtime", {}), "data.runtime", errors)
    data_runtime_mode = data_runtime.get("mode")
    if not isinstance(data_runtime_mode, str) or data_runtime_mode not in _DATA_RUNTIME_MODES:
        _err(
            errors,
            "data.runtime.mode must be one of: " + ", ".join(sorted(_DATA_RUNTIME_MODES)),
        )
    if data_runtime_mode == "local_jsonl":
        local_jsonl_path = data_runtime.get("local_jsonl_path")
        if not isinstance(local_jsonl_path, str) or not local_jsonl_path.strip():
            _err(
                errors,
                "data.runtime.local_jsonl_path must be a non-empty string when mode=local_jsonl",
            )
    if data_runtime_mode == "hf":
        datasets = data.get("datasets")
        if not isinstance(datasets, list) or not datasets:
            _err(errors, "data.datasets must be a non-empty list when data.runtime.mode=hf")
    num_workers = data.get("num_workers")
    if num_workers is not None:
        if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers <= 0:
            _err(errors, "data.num_workers must be null or a positive integer")

    max_total = _require_number(length_cfg, "max_total_tokens", errors)
    max_source = _require_number(length_cfg, "max_source_tokens", errors)
    max_output = _require_number(length_cfg, "max_output_tokens", errors)
    min_avail = _require_number(length_cfg, "min_available_output_tokens", errors)
    safety = _require_number(length_cfg, "safety_margin_tokens", errors, allow_zero=True)
    prompt_template_tokens = length_cfg.get("prompt_template_tokens", 0)
    if isinstance(prompt_template_tokens, bool) or not isinstance(prompt_template_tokens, (int, float)):
        _err(errors, "data.length.prompt_template_tokens must be numeric")
    elif prompt_template_tokens < 0:
        _err(errors, "data.length.prompt_template_tokens must be >= 0")

    if max_total is not None and max_length is not None and max_total > max_length:
        _err(errors, "data.length.max_total_tokens must be <= model.max_length")
    if (
        max_total is not None
        and isinstance(max_seq_length, (int, float))
        and max_total > float(max_seq_length)
    ):
        _err(errors, "data.length.max_total_tokens must be <= model.max_seq_length")

    prompt_tokens: float | None
    if isinstance(prompt_template_tokens, (int, float)) and not isinstance(
        prompt_template_tokens, bool
    ):
        prompt_tokens = float(prompt_template_tokens)
    else:
        prompt_tokens = None

    if None not in {max_source, min_avail, safety, max_total, prompt_tokens}:
        if (max_source + min_avail + safety + prompt_tokens) > max_total:
            _err(
                errors,
                "data.length.max_source_tokens + min_available_output_tokens + "
                "prompt_template_tokens + "
                "safety_margin_tokens must be <= data.length.max_total_tokens",
            )
    overflow = length_cfg.get("overflow")
    if not isinstance(overflow, str) or overflow not in _OVERFLOW_POLICIES:
        _err(
            errors,
            "data.length.overflow must be one of: split, skip, truncate",
        )
    tokenizer_fallback = length_cfg.get("tokenizer_fallback", "whitespace")
    if tokenizer_fallback not in {"whitespace", "error"}:
        _err(errors, "data.length.tokenizer_fallback must be 'whitespace' or 'error'")
    split_cfg = _as_dict(length_cfg.get("split", {}), "data.length.split", errors)
    split_unit = split_cfg.get("unit", "sentence")
    if not isinstance(split_unit, str) or split_unit not in _SPLIT_UNIT_POLICIES:
        _err(
            errors,
            "data.length.split.unit must be one of: "
            + ", ".join(sorted(_SPLIT_UNIT_POLICIES)),
        )
    max_chunks = split_cfg.get("max_chunks_per_row")
    if isinstance(max_chunks, bool) or not isinstance(max_chunks, int) or max_chunks <= 0:
        _err(errors, "data.length.split.max_chunks_per_row must be a positive integer")
    max_tokens_per_chunk = split_cfg.get("max_source_tokens_per_chunk")
    if (
        isinstance(max_tokens_per_chunk, bool)
        or not isinstance(max_tokens_per_chunk, int)
        or max_tokens_per_chunk <= 0
    ):
        _err(
            errors,
            "data.length.split.max_source_tokens_per_chunk must be a positive integer",
        )
    min_chunk_tokens = split_cfg.get("min_chunk_tokens")
    if (
        isinstance(min_chunk_tokens, bool)
        or not isinstance(min_chunk_tokens, int)
        or min_chunk_tokens <= 0
    ):
        _err(errors, "data.length.split.min_chunk_tokens must be a positive integer")
    if (
        isinstance(min_chunk_tokens, int)
        and isinstance(max_tokens_per_chunk, int)
        and min_chunk_tokens > max_tokens_per_chunk
    ):
        _err(
            errors,
            "data.length.split.min_chunk_tokens must be <= "
            "data.length.split.max_source_tokens_per_chunk",
        )
    fallback_for_long_sentence = split_cfg.get("fallback_for_long_sentence", "skip")
    if (
        not isinstance(fallback_for_long_sentence, str)
        or fallback_for_long_sentence not in _SPLIT_LONG_SENTENCE_POLICIES
    ):
        _err(
            errors,
            "data.length.split.fallback_for_long_sentence must be one of: "
            + ", ".join(sorted(_SPLIT_LONG_SENTENCE_POLICIES)),
        )
    on_max_chunks_exceeded = split_cfg.get("on_max_chunks_exceeded", "skip")
    if (
        not isinstance(on_max_chunks_exceeded, str)
        or on_max_chunks_exceeded not in _SPLIT_MAX_CHUNKS_EXCEEDED_POLICIES
    ):
        _err(
            errors,
            "data.length.split.on_max_chunks_exceeded must be one of: "
            + ", ".join(sorted(_SPLIT_MAX_CHUNKS_EXCEEDED_POLICIES)),
        )

    q1 = _as_dict(inference.get("q1", {}), "inference.q1", errors)
    q2 = _as_dict(inference.get("q2", {}), "inference.q2", errors)
    _require_number(q1, "max_new_tokens", errors)
    _require_number(q2, "max_new_tokens", errors)

    subset_cfg = _as_dict(pipeline.get("subset", {}), "pipeline.subset", errors)
    strategy = subset_cfg.get("strategy")
    if strategy not in {"fraction", "fixed_size"}:
        _err(errors, "pipeline.subset.strategy must be 'fraction' or 'fixed_size'")
    if strategy == "fraction":
        fraction = subset_cfg.get("fraction")
        if not isinstance(fraction, (int, float)) or not (0 < float(fraction) <= 1):
            _err(errors, "pipeline.subset.fraction must be in (0, 1]")
    elif strategy == "fixed_size":
        fixed_size = subset_cfg.get("fixed_size")
        if not isinstance(fixed_size, int) or fixed_size <= 0:
            _err(errors, "pipeline.subset.fixed_size must be a positive integer")
    min_size = subset_cfg.get("min_size")
    if not isinstance(min_size, int) or min_size <= 0:
        _err(errors, "pipeline.subset.min_size must be a positive integer")
    max_size = subset_cfg.get("max_size")
    if max_size is not None:
        if not isinstance(max_size, int) or max_size <= 0:
            _err(errors, "pipeline.subset.max_size must be null or a positive integer")
        elif isinstance(min_size, int) and max_size < min_size:
            _err(errors, "pipeline.subset.max_size must be >= min_size")

    subset_size = data.get("subset_size")
    if subset_size is not None and (not isinstance(subset_size, int) or subset_size <= 0):
        _err(errors, "data.subset_size must be null or a positive integer")

    eval_after = _as_dict(
        pipeline.get("eval_after_subset", {}), "pipeline.eval_after_subset", errors
    )
    every_n = eval_after.get("every_n_subsets")
    if not isinstance(every_n, int) or every_n <= 0:
        _err(errors, "pipeline.eval_after_subset.every_n_subsets must be > 0")
    stage_cfg = _as_dict(pipeline.get("stage", {}), "pipeline.stage", errors)
    max_subsets = stage_cfg.get("max_subsets")
    if max_subsets is not None and (isinstance(max_subsets, bool) or not isinstance(max_subsets, int) or max_subsets <= 0):
        _err(errors, "pipeline.stage.max_subsets must be null or a positive integer")
    use_sampled_data = stage_cfg.get("use_sampled_data")
    if use_sampled_data is not None and not isinstance(use_sampled_data, bool):
        _err(errors, "pipeline.stage.use_sampled_data must be a boolean")
    subset_archive_cfg = _as_dict(
        stage_cfg.get("subset_archive", {}),
        "pipeline.stage.subset_archive",
        errors,
    )
    subset_archive_enabled = subset_archive_cfg.get("enabled")
    if subset_archive_enabled is not None and not isinstance(subset_archive_enabled, bool):
        _err(errors, "pipeline.stage.subset_archive.enabled must be a boolean")
    subset_archive_format = subset_archive_cfg.get("format")
    if subset_archive_format is not None and (
        not isinstance(subset_archive_format, str)
        or subset_archive_format not in _SUBSET_ARCHIVE_FORMATS
    ):
        _err(
            errors,
            "pipeline.stage.subset_archive.format must be one of: "
            + ", ".join(sorted(_SUBSET_ARCHIVE_FORMATS)),
        )
    subset_archive_output_dir = subset_archive_cfg.get("output_dir")
    if subset_archive_output_dir is not None and (
        not isinstance(subset_archive_output_dir, str)
        or not subset_archive_output_dir.strip()
    ):
        _err(errors, "pipeline.stage.subset_archive.output_dir must be a non-empty string")
    subset_archive_delete = subset_archive_cfg.get("delete_original_after_archive")
    if subset_archive_delete is not None and not isinstance(subset_archive_delete, bool):
        _err(
            errors,
            "pipeline.stage.subset_archive.delete_original_after_archive must be a boolean",
        )

    if training.get("backend") != "unsloth":
        _err(errors, "training.backend must be 'unsloth'")
    _validate_training_runtime(training, errors)
    base_update = _as_dict(training.get("base_update", {}), "training.base_update", errors)
    if base_update.get("mode") not in {"lora", "full_weight"}:
        _err(errors, "training.base_update.mode must be 'lora' or 'full_weight'")
    lora_cfg = _as_dict(base_update.get("lora", {}), "training.base_update.lora", errors)
    target_modules = lora_cfg.get("target_modules")
    if target_modules is not None:
        if isinstance(target_modules, str):
            if not target_modules.strip():
                _err(errors, "training.base_update.lora.target_modules must be non-empty")
        elif isinstance(target_modules, list):
            if not target_modules:
                _err(errors, "training.base_update.lora.target_modules list must be non-empty")
            for idx, module_name in enumerate(target_modules):
                if not isinstance(module_name, str) or not module_name.strip():
                    _err(
                        errors,
                        f"training.base_update.lora.target_modules[{idx}] must be a non-empty string",
                    )
        else:
            _err(
                errors,
                "training.base_update.lora.target_modules must be either a string or a list of strings",
            )

    local_logging = _as_dict(logging_cfg.get("local", {}), "logging.local", errors)
    for key in ("enabled", "write_effective_config", "write_config_hash"):
        if not isinstance(local_logging.get(key), bool):
            _err(errors, f"logging.local.{key} must be a boolean")
    root_dir = local_logging.get("root_dir")
    if not isinstance(root_dir, str) or not root_dir.strip():
        _err(errors, "logging.local.root_dir must be a non-empty string")

    required_fields = logging_cfg.get("required_event_fields")
    if not isinstance(required_fields, list):
        _err(errors, "logging.required_event_fields must be a list")
    else:
        missing = [field for field in _REQUIRED_LOG_FIELDS if field not in required_fields]
        if missing:
            _err(
                errors,
                "logging.required_event_fields must include " + ", ".join(missing),
            )

    _validate_external_api_env_names(external_api, errors)
    _validate_subprocess_command(
        "inference",
        inference,
        "mode",
        "command",
        _INFERENCE_RUNTIME_MODES,
        errors,
    )
    _validate_subprocess_command(
        "qe",
        _as_dict(cfg.get("qe", {}), "qe", errors),
        "mode",
        "command",
        _QE_RUNTIME_MODES,
        errors,
    )
    _validate_subprocess_command(
        "external_api",
        external_api,
        "mode",
        "command",
        _API_RUNTIME_MODES,
        errors,
    )

    if errors:
        raise ConfigValidationError("Config validation failed:\n- " + "\n- ".join(errors))
