"""Mock inference worker for subprocess runtime integration tests."""

from __future__ import annotations

from typing import Any

from scp_stage4.data import read_jsonl, write_jsonl
from scp_stage4.pipeline.workers.common import (
    parse_worker_args,
    validate_phase_request_rows,
    validate_phase_response_rows,
)


def _build_translation(row: dict[str, Any]) -> str:
    q_tag = str(row.get("q_tag", "q1"))
    row_id = str(row.get("row_id", row.get("id", "unknown")))
    if q_tag == "q2":
        return f"KO_Q2::{row_id}"
    return f"KO_Q1::{row_id}"


def main(argv: list[str] | None = None) -> int:
    args = parse_worker_args(description="Mock inference worker", argv=argv)

    requests = read_jsonl(args.input_path)
    schema = validate_phase_request_rows(requests, args=args, context="inference")
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
    validate_phase_response_rows(responses, schema=schema, context="inference")
    write_jsonl(args.output_path, responses, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
