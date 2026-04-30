"""Report per-dataset source ratios from prepared JSONL artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


def _compute_counts(path: Path) -> tuple[int, Counter[str]]:
    total = 0
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            dataset = row.get("dataset")
            if not isinstance(dataset, str) or not dataset.strip():
                dataset = "<unknown>"
            counts[dataset] += 1
            total += 1
    return total, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show per-dataset source ratios")
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts/data",
        help="Directory containing prepared JSONL artifacts",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=[
            "datapool.train.jsonl",
            "datapool.eval.jsonl",
            "datapool.normalized.jsonl",
        ],
        help="Artifact filenames to report",
    )
    args = parser.parse_args(argv)

    artifacts_dir = Path(args.artifacts_dir)
    had_any = False
    for name in args.files:
        path = artifacts_dir / name
        if not path.exists():
            continue
        had_any = True
        total, counts = _compute_counts(path)
        print(f"{name}\ttotal={total}")
        if total <= 0:
            continue
        for dataset, value in counts.most_common():
            ratio = (value / total) * 100.0
            print(f"{dataset}\t{value}\t{ratio:.2f}%")
        print("")

    if not had_any:
        print(
            f"No artifact files found under {artifacts_dir} for requested names: {', '.join(args.files)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
