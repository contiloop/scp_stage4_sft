"""Mock external API worker for subprocess runtime integration tests."""

from __future__ import annotations

from typing import Any

from scp_stage4.data import read_jsonl, write_jsonl
from scp_stage4.pipeline.workers.common import (
    parse_worker_args,
    validate_phase_request_rows,
    validate_phase_response_rows,
    WorkerContractError,
)


def _gold_for_row(row: dict[str, Any]) -> str:
    row_id = str(row.get("row_id", "unknown"))
    return f"KO_GOLD::{row_id}"


def main(argv: list[str] | None = None) -> int:
    args = parse_worker_args(description="Mock API worker", argv=argv)

    requests = read_jsonl(args.input_path)
    schema = validate_phase_request_rows(requests, args=args, context="external_api")
    responses = []
    for row in requests:
        req_id = str(row.get("request_id", ""))
        responses.append(
            {
                "request_id": req_id,
                "status": "ok",
                "gold": _gold_for_row(dict(row)),
                "teacher_label": "minor_edit",
                "usage": {
                    "input_tokens": 96,
                    "output_tokens": 72,
                    "total_tokens": 168,
                },
                "cost": {
                    "currency": "USD",
                    "estimated": 0.0,
                },
                "latency_ms": 1.0,
                "attempt": 1,
                "reason": None,
                "error": None,
            }
        )

    validate_phase_response_rows(responses, schema=schema, context="external_api")
    for idx, row in enumerate(responses):
        status = str(row.get("status", "ok"))
        if status == "ok":
            gold = row.get("gold")
            if not isinstance(gold, str) or not gold.strip():
                raise WorkerContractError(
                    f"external_api response row {idx} missing non-empty gold for status=ok"
                )
    write_jsonl(args.output_path, responses, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
