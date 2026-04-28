"""Remote-runtime skeleton checks (no real GPU/QE/API execution)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from scp_stage4.config.loader import compose_config
from scp_stage4.config.validator import validate_config
from scp_stage4.pipeline.smoke_local import run_smoke


def _is_placeholder_model(model_name: str) -> bool:
    lowered = model_name.strip().lower()
    return lowered in {"", "placeholder", "gpt-5.4"}


def cmd_validate_env(config_path: str, overrides: list[str]) -> int:
    cfg = compose_config(config_path, overrides=overrides)
    validate_config(cfg)

    primary = cfg["external_api"]["primary"]
    env_name = primary["api_key_env"]
    present = env_name in os.environ and bool(os.environ.get(env_name))

    print(f"remote-env: config OK, api_key_env={env_name}, present={present}")
    return 0


def cmd_smoke_qe(config_path: str, overrides: list[str]) -> int:
    cfg = compose_config(config_path, overrides=overrides)
    validate_config(cfg)

    isolation_env = cfg.get("qe", {}).get("isolation", {}).get("env", {})
    required = [
        isolation_env.get("comet_python_env", "COMET_PYTHON"),
        isolation_env.get("metricx_python_env", "METRICX_PYTHON"),
    ]

    missing = [name for name in required if not os.environ.get(str(name))]
    if missing:
        print(
            "smoke-remote-qe: missing env vars: " + ", ".join(map(str, missing)),
            file=sys.stderr,
        )
        return 1

    for env_name in required:
        value = os.environ.get(str(env_name), "")
        if value and not Path(value).exists():
            print(
                f"smoke-remote-qe: {env_name} points to missing path: {value}",
                file=sys.stderr,
            )
            return 1

    print("smoke-remote-qe: env contracts OK")
    return 0


def cmd_smoke_model(config_path: str, overrides: list[str]) -> int:
    cfg = compose_config(config_path, overrides=overrides)
    validate_config(cfg)

    backend = cfg.get("training", {}).get("backend")
    if backend != "unsloth":
        print("smoke-remote-model: training.backend must be unsloth", file=sys.stderr)
        return 1

    print("smoke-remote-model: config contract OK (no GPU load performed)")
    return 0


def cmd_smoke_api(config_path: str, overrides: list[str]) -> int:
    cfg = compose_config(config_path, overrides=overrides)
    validate_config(cfg)

    model_name = str(cfg.get("external_api", {}).get("primary", {}).get("model", ""))
    if _is_placeholder_model(model_name):
        print(
            "smoke-remote-api: external_api.primary.model is placeholder; set a real model",
            file=sys.stderr,
        )
        return 1

    env_name = str(cfg["external_api"]["primary"]["api_key_env"])
    if not os.environ.get(env_name):
        print(
            f"smoke-remote-api: required env var missing: {env_name}",
            file=sys.stderr,
        )
        return 1

    print("smoke-remote-api: config/env contracts OK (no real API call performed)")
    return 0


def cmd_dry_run_subset(config_path: str, overrides: list[str]) -> int:
    merged_overrides = list(overrides) + ["data.subset_size=32"]
    summary = run_smoke(
        config_path=config_path,
        overrides=merged_overrides,
        run_id_override="dry_run_remote_subset",
        subset_size_override=32,
    )
    print(
        "dry-run-remote-subset: generated local mock artifacts "
        f"at {summary['run_root']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Remote skeleton checks")
    parser.add_argument(
        "command",
        choices=[
            "validate-env",
            "smoke-qe",
            "smoke-model",
            "smoke-api",
            "dry-run-subset",
        ],
    )
    parser.add_argument("--config", default="configs/scp_stage4.yaml")
    args, overrides = parser.parse_known_args(argv)

    if args.command == "validate-env":
        return cmd_validate_env(args.config, overrides)
    if args.command == "smoke-qe":
        return cmd_smoke_qe(args.config, overrides)
    if args.command == "smoke-model":
        return cmd_smoke_model(args.config, overrides)
    if args.command == "smoke-api":
        return cmd_smoke_api(args.config, overrides)
    if args.command == "dry-run-subset":
        return cmd_dry_run_subset(args.config, overrides)

    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
