"""CLI: compose and validate config for local harness."""

from __future__ import annotations

import argparse
import sys

from scp_stage4.config.loader import ConfigLoadError, compose_config
from scp_stage4.config.validator import ConfigValidationError, validate_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate composed SCP Stage 4 config")
    parser.add_argument("--config", default="configs/scp_stage4.yaml")
    args, overrides = parser.parse_known_args(argv)

    try:
        cfg = compose_config(args.config, overrides=overrides)
        validate_config(cfg)
    except (ConfigLoadError, ConfigValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("validate-config: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
