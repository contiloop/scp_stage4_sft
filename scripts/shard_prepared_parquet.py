#!/usr/bin/env python3
"""Shard prepared train/eval parquet artifacts into Hub-friendly part files."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def _write_shards(
    *,
    source_path: Path,
    out_dir: Path,
    rows_per_shard: int,
    row_group_size: int,
) -> dict[str, Any]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = pq.ParquetFile(str(source_path))
    shard_idx = 0
    total_rows = 0
    total_size = 0
    files: list[str] = []
    pending: list[pa.Table] = []
    pending_rows = 0

    def flush() -> None:
        nonlocal shard_idx, pending_rows, total_size
        if not pending:
            return
        table = pa.concat_tables(pending, promote_options="default")
        pending.clear()
        pending_rows = 0
        shard_path = out_dir / f"part-{shard_idx:05d}.parquet"
        pq.write_table(
            table,
            shard_path,
            compression="zstd",
            row_group_size=row_group_size,
        )
        files.append(f"{out_dir.name}/{shard_path.name}")
        total_size += shard_path.stat().st_size
        shard_idx += 1

    for batch in parquet_file.iter_batches(batch_size=row_group_size):
        table = pa.Table.from_batches([batch])
        pending.append(table)
        pending_rows += table.num_rows
        total_rows += table.num_rows
        if pending_rows >= rows_per_shard:
            flush()
    flush()

    return {
        "rows": total_rows,
        "num_shards": shard_idx,
        "size_bytes": total_size,
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bundle-tag", required=True)
    parser.add_argument("--rows-per-train-shard", type=int, default=512_000)
    parser.add_argument("--rows-per-eval-shard", type=int, default=512_000)
    parser.add_argument("--row-group-size", type=int, default=4096)
    args = parser.parse_args()

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train = _write_shards(
        source_path=args.artifacts_dir / "datapool.train.parquet",
        out_dir=args.out_dir / "train",
        rows_per_shard=args.rows_per_train_shard,
        row_group_size=args.row_group_size,
    )
    eval_ = _write_shards(
        source_path=args.artifacts_dir / "datapool.eval.parquet",
        out_dir=args.out_dir / "eval",
        rows_per_shard=args.rows_per_eval_shard,
        row_group_size=args.row_group_size,
    )

    manifest = {
        "bundle_tag": args.bundle_tag,
        "train": train,
        "eval": eval_,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
