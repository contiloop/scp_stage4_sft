"""Mock external API worker for subprocess runtime integration tests."""

from __future__ import annotations

import argparse
from typing import Any

from scp_stage4.data import read_jsonl, write_jsonl


def _gold_for_row(row: dict[str, Any]) -> str:
    row_id = str(row.get("row_id", "unknown"))
    return f"KO_GOLD::{row_id}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mock API worker")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    requests = read_jsonl(args.input)
    responses = []
    for row in requests:
        req_id = str(row.get("request_id", ""))
        responses.append(
            {
                "request_id": req_id,
                "status": "ok",
                "gold": _gold_for_row(dict(row)),
                "teacher_label": "minor_edit",
                "error": None,
            }
        )

    write_jsonl(args.output, responses, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
