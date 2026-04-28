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
_INFERENCE_RUNTIME_MODES = {"mock", "subprocess"}
_QE_RUNTIME_MODES = {"mock", "subprocess"}
_API_RUNTIME_MODES = {"mock", "subprocess"}


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
    max_total = _require_number(length_cfg, "max_total_tokens", errors)
    max_source = _require_number(length_cfg, "max_source_tokens", errors)
    max_output = _require_number(length_cfg, "max_output_tokens", errors)
    min_avail = _require_number(length_cfg, "min_available_output_tokens", errors)
    safety = _require_number(length_cfg, "safety_margin_tokens", errors, allow_zero=True)

    if max_total is not None and max_length is not None and max_total > max_length:
        _err(errors, "data.length.max_total_tokens must be <= model.max_length")
    if (
        max_total is not None
        and isinstance(max_seq_length, (int, float))
        and max_total > float(max_seq_length)
    ):
        _err(errors, "data.length.max_total_tokens must be <= model.max_seq_length")

    if None not in {max_source, min_avail, safety, max_total}:
        if (max_source + min_avail + safety) > max_total:
            _err(
                errors,
                "data.length.max_source_tokens + min_available_output_tokens + "
                "safety_margin_tokens must be <= data.length.max_total_tokens",
            )
    overflow = length_cfg.get("overflow")
    if not isinstance(overflow, str) or overflow not in _OVERFLOW_POLICIES:
        _err(
            errors,
            "data.length.overflow must be one of: split, skip, truncate",
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

    if training.get("backend") != "unsloth":
        _err(errors, "training.backend must be 'unsloth'")
    base_update = _as_dict(training.get("base_update", {}), "training.base_update", errors)
    if base_update.get("mode") not in {"lora", "full_weight"}:
        _err(errors, "training.base_update.mode must be 'lora' or 'full_weight'")

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
