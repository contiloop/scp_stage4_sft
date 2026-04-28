"""Mock training worker for subprocess runtime integration tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from scp_stage4.data import read_jsonl, write_jsonl
from scp_stage4.pipeline.workers.common import (
    parse_worker_args,
    validate_phase_request_rows,
    validate_phase_response_rows,
    WorkerContractError,
)


def _phase(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "update-base"
    return str(rows[0].get("phase", "update-base"))


def main(argv: list[str] | None = None) -> int:
    args = parse_worker_args(description="Mock training worker", argv=argv)

    requests = [dict(row) for row in read_jsonl(args.input_path)]
    phase = str(args.phase or _phase(requests))
    if requests:
        request_phase = str(requests[0].get("phase", phase))
        if request_phase != phase:
            raise WorkerContractError(
                f"training request phase mismatch: cli phase={phase!r}, row phase={request_phase!r}"
            )

    if phase == "train-collapse-lora":
        schema = validate_phase_request_rows(requests, args=args, context="training.collapse")
        adapter_path = str(
            requests[0].get("adapter_path", "collapse_adapter") if requests else "collapse_adapter"
        )
        Path(adapter_path).mkdir(parents=True, exist_ok=True)
        responses = [
            {
                "status": "ok",
                "adapter_path": adapter_path,
                "trained_rows": len(requests),
                "backend": "mock_subprocess",
                "error": None,
            }
        ]
    elif phase == "unload-collapse-lora":
        schema = validate_phase_request_rows(requests, args=args, context="training.unload")
        adapter_path = requests[0].get("adapter_path") if requests else None
        adapter_hash = hashlib.sha256(str(adapter_path).encode("utf-8")).hexdigest()
        responses = [
            {
                "status": "ok",
                "adapter_path": adapter_path,
                "clean_base": True,
                "active_adapters": [],
                "collapse_merged": False,
                "adapter_registry_hash": adapter_hash,
                "verified_adapter_path": adapter_path,
                "backend": "mock_subprocess",
                "error": None,
            }
        ]
    else:
        schema = validate_phase_request_rows(requests, args=args, context="training.update")
        output_dir = str(
            requests[0].get("output_dir", "train_final") if requests else "train_final"
        )
        checkpoint_path = str(Path(output_dir) / "main_adapter")
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        responses = [
            {
                "status": "ok",
                "checkpoint_path": checkpoint_path,
                "trained_rows": len(requests),
                "backend": "mock_subprocess",
                "error": None,
            }
        ]

    validate_phase_response_rows(responses, schema=schema, context="training")
    write_jsonl(args.output_path, responses, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
