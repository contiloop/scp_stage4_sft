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
        pass

    # Makefile passes overrides through a shell expansion step, which can strip
    # quotes inside list/dict literals (e.g. ["a","b"] -> [a,b]). For container-like
    # values, accept YAML-style parsing as a compatibility fallback.
    text = raw.strip()
    if text.startswith("[") or text.startswith("{"):
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError:
            yaml = None  # type: ignore[assignment]
        if yaml is not None:
            try:
                parsed = yaml.safe_load(text)
            except Exception:
                parsed = None
            if isinstance(parsed, (list, dict)):
                return parsed

    # Last-resort compatibility for shell-expanded list literals such as
    # [python3,-m,scp_stage4.pipeline.workers.inference_worker].
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        parts = [part.strip() for part in inner.split(",")]
        if all(parts):
            return parts
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


def _looks_like_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _yaml_quote_string(value: str) -> str:
    if value == "":
        return "''"

    lowered = value.lower()
    if lowered in {"null", "~", "true", "false"}:
        return json.dumps(value, ensure_ascii=False)
    if _looks_like_number(value):
        return json.dumps(value, ensure_ascii=False)

    special_chars = set(":#{}[]&*?!|>%'\"`")
    if (
        value.strip() != value
        or any(ch in special_chars for ch in value)
        or value.startswith(("-", "@", "`", " "))
        or "\n" in value
        or "\r" in value
        or "\t" in value
    ):
        return json.dumps(value, ensure_ascii=False)

    return value


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return _yaml_quote_string(value)
    return json.dumps(value, ensure_ascii=False)


def _render_yaml_lines(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent

    if isinstance(value, dict):
        if not value:
            return [f"{prefix}{{}}"]

        lines: list[str] = []
        for key, item in value.items():
            key_text = _yaml_quote_string(str(key))
            if isinstance(item, (dict, list)):
                if isinstance(item, dict) and not item:
                    lines.append(f"{prefix}{key_text}: {{}}")
                elif isinstance(item, list) and not item:
                    lines.append(f"{prefix}{key_text}: []")
                else:
                    lines.append(f"{prefix}{key_text}:")
                    lines.extend(_render_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{key_text}: {_yaml_scalar(item)}")
        return lines

    if isinstance(value, list):
        if not value:
            return [f"{prefix}[]"]

        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                if isinstance(item, dict) and not item:
                    lines.append(f"{prefix}- {{}}")
                elif isinstance(item, list) and not item:
                    lines.append(f"{prefix}- []")
                else:
                    lines.append(f"{prefix}-")
                    lines.extend(_render_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return lines

    return [f"{prefix}{_yaml_scalar(value)}"]


def _serialize_as_yaml_text(value: Any) -> str:
    try:
        import yaml  # type: ignore

        return yaml.safe_dump(
            value,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
    except ModuleNotFoundError:
        return "\n".join(_render_yaml_lines(value)) + "\n"


def save_effective_config(cfg: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_serialize_as_yaml_text(cfg), encoding="utf-8")
