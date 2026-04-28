"""Mock QE worker for subprocess runtime integration tests."""

from __future__ import annotations

import argparse
import hashlib
from typing import Any

from scp_stage4.data import read_jsonl, write_jsonl


def _stable_fraction(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    bucket = int(digest[:6], 16) % 1000
    return bucket / 1000.0


def _score_for_row(row: dict[str, Any]) -> float:
    row_id = str(row.get("row_id", row.get("id", "unknown")))
    base = 0.10 + _stable_fraction(row_id) * 0.30
    q_tag = str(row.get("q_tag", "q1"))
    if q_tag == "q2":
        return round(base + 0.05, 6)
    return round(base, 6)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mock QE worker")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    requests = read_jsonl(args.input)
    responses = []
    for row in requests:
        req_id = str(row.get("id", ""))
        backend = str(row.get("backend", "metricx24"))
        responses.append(
            {
                "id": req_id,
                "score": _score_for_row(dict(row)),
                "backend": backend,
                "model_name": f"mock/{backend}",
                "runtime_ms": 1.0,
                "status": "ok",
                "error": None,
            }
        )

    write_jsonl(args.output, responses, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
