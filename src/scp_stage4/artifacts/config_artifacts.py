"""Helpers for effective config and config hash artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

REDACTED = "[REDACTED]"
_SECRET_KEY_PATTERN = re.compile(
    r"(^|_)(api_key|auth_token|access_token|refresh_token|token|secret|password|passphrase|private_key|access_key|authorization)($|_)",
    re.IGNORECASE,
)
_SECRET_EXACT_ALLOWLIST = {"api_key_env", "token_env", "secret_env", "password_env", "auth_env"}
_OPENAI_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-~+/]+=*\b")


def _is_secret_key(key: str) -> bool:
    key_lower = key.lower().strip()
    if key_lower in _SECRET_EXACT_ALLOWLIST:
        return False
    if key_lower.endswith("_env"):
        return False
    return bool(_SECRET_KEY_PATTERN.search(key_lower))


def _redact_string_if_secret_like(value: str) -> str:
    if _OPENAI_KEY_PATTERN.search(value):
        return _OPENAI_KEY_PATTERN.sub(REDACTED, value)
    if _BEARER_PATTERN.search(value):
        return _BEARER_PATTERN.sub("Bearer [REDACTED]", value)
    return value


def redact_config_secrets(value: Any) -> Any:
    """Return a deep-copied config with secret-like fields redacted."""
    copied = copy.deepcopy(value)
    return _redact_config_node(copied)


def _redact_config_node(value: Any, parent_key: str | None = None) -> Any:
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if _is_secret_key(key_str):
                output[key_str] = REDACTED
            else:
                output[key_str] = _redact_config_node(item, parent_key=key_str)
        return output
    if isinstance(value, list):
        return [_redact_config_node(item, parent_key=parent_key) for item in value]
    if isinstance(value, tuple):
        return [_redact_config_node(item, parent_key=parent_key) for item in value]
    if isinstance(value, str):
        if parent_key and _is_secret_key(parent_key):
            return REDACTED
        return _redact_string_if_secret_like(value)
    return value


def compute_config_hash(effective_config: Mapping[str, Any] | Any) -> str:
    """Compute deterministic SHA256 hash from a secret-redacted config object."""
    redacted = redact_config_secrets(effective_config)
    canonical = json.dumps(redacted, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _serialize_as_yaml_text(redacted_config: Any) -> str:
    try:
        import yaml  # type: ignore

        return yaml.safe_dump(redacted_config, sort_keys=True, allow_unicode=True)
    except Exception:
        # JSON is valid YAML 1.2 and keeps this helper dependency-light.
        return json.dumps(redacted_config, ensure_ascii=False, indent=2, sort_keys=True)


def persist_effective_config_artifacts(
    *,
    run_dir: str | Path,
    effective_config: Mapping[str, Any] | Any,
    write_effective_config: bool = True,
    write_config_hash: bool = True,
) -> dict[str, Any]:
    """
    Persist redacted effective config and config hash.

    Returns:
      {
        "effective_config_path": Path | None,
        "config_hash_path": Path | None,
        "config_hash": str,
      }
    """
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    redacted = redact_config_secrets(effective_config)
    config_hash = compute_config_hash(redacted)

    effective_config_path: Path | None = None
    if write_effective_config:
        effective_config_path = run_path / "effective_config.yaml"
        effective_config_path.write_text(_serialize_as_yaml_text(redacted), encoding="utf-8")

    config_hash_path: Path | None = None
    if write_config_hash:
        config_hash_path = run_path / "config_hash.txt"
        config_hash_path.write_text(f"{config_hash}\n", encoding="utf-8")

    return {
        "effective_config_path": effective_config_path,
        "config_hash_path": config_hash_path,
        "config_hash": config_hash,
    }
