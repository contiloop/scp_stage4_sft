"""Backward-compatible wrapper for JSONL validation CLI."""

from __future__ import annotations

from scp_stage4.schema.validate_jsonl import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
