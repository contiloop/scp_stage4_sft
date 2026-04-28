"""Mock inference worker for subprocess runtime integration tests."""

from __future__ import annotations

import argparse
from typing import Any

from scp_stage4.data import read_jsonl, write_jsonl


def _build_translation(row: dict[str, Any]) -> str:
    q_tag = str(row.get("q_tag", "q1"))
    row_id = str(row.get("row_id", row.get("id", "unknown")))
    if q_tag == "q2":
        return f"KO_Q2::{row_id}"
    return f"KO_Q1::{row_id}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mock inference worker")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    requests = read_jsonl(args.input)
    responses = []
    for row in requests:
        req_id = str(row.get("id", ""))
        responses.append(
            {
                "id": req_id,
                "status": "ok",
                "mt": _build_translation(dict(row)),
                "error": None,
            }
        )
    write_jsonl(args.output, responses, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
