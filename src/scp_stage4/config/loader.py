"""Compose split configuration files into one effective config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class ConfigLoadError(RuntimeError):
    """Raised when config files cannot be loaded or composed."""


def _try_parse_json_or_yaml(text: str, source: Path) -> dict[str, Any]:
    text = text.strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        raise ConfigLoadError(f"Config must be an object: {source}")
    except json.JSONDecodeError:
        pass

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ConfigLoadError(
            f"{source} is not JSON-compatible and PyYAML is not installed"
        ) from exc

    parsed_yaml = yaml.safe_load(text)
    if parsed_yaml is None:
        return {}
    if not isinstance(parsed_yaml, dict):
        raise ConfigLoadError(f"Config must be an object: {source}")
    return parsed_yaml


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigLoadError(f"Missing config file: {path}")
    return _try_parse_json_or_yaml(path.read_text(encoding="utf-8"), path)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _parse_override_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _set_by_dotpath(target: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cursor = target
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def apply_overrides(cfg: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    out = dict(cfg)
    for item in overrides:
        if "=" not in item:
            raise ConfigLoadError(
                f"Invalid override '{item}'. Expected key=value format."
            )
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigLoadError(f"Invalid override '{item}'. Empty key.")
        _set_by_dotpath(out, key, _parse_override_value(raw_value.strip()))
    return out


def _normalize_defaults_item(item: Any) -> str:
    if isinstance(item, str):
        return item
    raise ConfigLoadError(
        "Only string defaults entries are supported in the local harness"
    )


def _resolve_config_path(config_dir: Path, default_name: str) -> Path:
    maybe = Path(default_name)
    if maybe.suffix:
        return config_dir / maybe
    return config_dir / f"{default_name}.yaml"


def compose_config(
    config_path: str | Path,
    overrides: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compose split config files into one dict.

    The root file must contain a `defaults` array with child file names.
    """

    root_path = Path(config_path)
    root = _read_config_file(root_path)

    defaults = root.get("defaults")
    if defaults is None:
        raise ConfigLoadError(f"Root config must define defaults: {root_path}")
    if not isinstance(defaults, list):
        raise ConfigLoadError(f"defaults must be a list: {root_path}")

    composed: dict[str, Any] = {}
    config_dir = root_path.parent

    for entry in defaults:
        name = _normalize_defaults_item(entry)
        child_path = _resolve_config_path(config_dir, name)
        child_cfg = _read_config_file(child_path)
        composed = _deep_merge(composed, child_cfg)

    for key, value in root.items():
        if key == "defaults":
            continue
        composed[key] = value

    if overrides:
        composed = apply_overrides(composed, overrides)

    model = composed.get("model", {})
    if isinstance(model, dict):
        if model.get("max_seq_length") is None and model.get("max_length") is not None:
            model["max_seq_length"] = model["max_length"]
            composed["model"] = model

    return composed


def save_effective_config(cfg: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
