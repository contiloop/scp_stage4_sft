"""CLI: validate JSONL artifacts and fixture shape for local contract checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

from scp_stage4.config.loader import compose_config
from scp_stage4.pipeline.io_utils import iter_jsonl

_ARTIFACT_NAMES = {
    "normalized",
    "input",
    "q1",
    "q2",
    "scored",
    "selected",
    "api_requests",
    "api",
    "train",
}


def _collect_jsonl_paths(run_root: Path) -> list[Path]:
    paths: list[Path] = []
    fixture_dir = Path("tests/fixtures")
    if fixture_dir.exists():
        paths.extend(sorted(fixture_dir.rglob("*.jsonl")))

    if run_root.exists():
        paths.extend(sorted(run_root.rglob("*.jsonl")))

    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _basic_validate_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(iter_jsonl(path), start=1):
        if "id" in row and not isinstance(row["id"], str):
            raise ValueError(f"{path}:{idx} id must be string")
        rows.append(dict(row))
    return rows


def _resolve_data_helpers() -> tuple[Callable[[str | Path], list[dict[str, Any]]] | None, Callable[..., None] | None]:
    try:
        from scp_stage4.data import read_jsonl, validate_row_id_preservation
    except ModuleNotFoundError:
        return None, None
    return read_jsonl, validate_row_id_preservation


def _resolve_schema_helper() -> Callable[[Iterable[dict[str, Any]], str], list[dict[str, Any]]] | None:
    try:
        from scp_stage4.schema import validate_artifact_rows
    except ModuleNotFoundError:
        return None
    return validate_artifact_rows


def _artifact_name_from_path(path: Path) -> str | None:
    if path.name == "train_rows.jsonl":
        return "train"
    if path.name in {
        "datapool.normalized.jsonl",
        "datapool.train.jsonl",
        "datapool.eval.jsonl",
        "datapool.train.sampled.jsonl",
    }:
        return "normalized"

    stem = path.stem
    if "." in stem:
        stem = stem.split(".", 1)[0]

    if stem in _ARTIFACT_NAMES:
        return stem
    return None


def _load_rows(path: Path, read_jsonl: Callable[[str | Path], list[dict[str, Any]]] | None) -> list[dict[str, Any]]:
    if read_jsonl is None:
        return _basic_validate_jsonl(path)
    rows = read_jsonl(path)
    return [dict(row) for row in rows]


def _validate_with_schema(
    path: Path,
    rows: list[dict[str, Any]],
    validate_artifact_rows: Callable[[Iterable[dict[str, Any]], str], list[dict[str, Any]]] | None,
) -> None:
    artifact = _artifact_name_from_path(path)
    if artifact is None:
        return
    if validate_artifact_rows is None:
        return
    validate_artifact_rows(rows, artifact)


def _row_id_chain_from_fixture_dir(fixture_dir: Path) -> list[tuple[Path, Path, bool, str, str]]:
    chain: list[tuple[Path, Path, bool, str, str]] = []
    mapping = {
        "input": fixture_dir / "input.happy.jsonl",
        "q1": fixture_dir / "q1.happy.jsonl",
        "q2": fixture_dir / "q2.happy.jsonl",
        "scored": fixture_dir / "scored.happy.jsonl",
        "selected": fixture_dir / "selected.happy.jsonl",
        "api_requests": fixture_dir / "api_requests.happy.jsonl",
        "api": fixture_dir / "api.happy.jsonl",
        "train": fixture_dir / "train.happy.jsonl",
    }
    if all(path.exists() for path in mapping.values()):
        chain.extend(
            [
                (mapping["input"], mapping["q1"], False, "input", "q1"),
                (mapping["q1"], mapping["q2"], False, "q1", "q2"),
                (mapping["q2"], mapping["scored"], False, "q2", "scored"),
                (mapping["scored"], mapping["selected"], True, "scored", "selected"),
                (
                    mapping["selected"],
                    mapping["api_requests"],
                    True,
                    "selected",
                    "api_requests",
                ),
                (mapping["api_requests"], mapping["api"], True, "api_requests", "api"),
                (mapping["api"], mapping["train"], True, "api", "train"),
            ]
        )
    return chain


def _row_id_chain_from_run_root(run_root: Path) -> list[tuple[Path, Path, bool, str, str]]:
    chain: list[tuple[Path, Path, bool, str, str]] = []
    subset_roots = sorted((run_root / "subsets").glob("subset_*")) if (run_root / "subsets").exists() else []
    for subset_root in subset_roots:
        mapping = {
            "input": subset_root / "input.jsonl",
            "q1": subset_root / "q1.jsonl",
            "q2": subset_root / "q2.jsonl",
            "scored": subset_root / "scored.jsonl",
            "selected": subset_root / "selected.jsonl",
            "api_requests": subset_root / "api_requests.jsonl",
            "api": subset_root / "api.jsonl",
            "train": subset_root / "train_final" / "train_rows.jsonl",
        }
        if not all(path.exists() for path in mapping.values()):
            continue
        chain.extend(
            [
                (mapping["input"], mapping["q1"], False, "input", "q1"),
                (mapping["q1"], mapping["q2"], False, "q1", "q2"),
                (mapping["q2"], mapping["scored"], False, "q2", "scored"),
                (mapping["scored"], mapping["selected"], True, "scored", "selected"),
                (
                    mapping["selected"],
                    mapping["api_requests"],
                    True,
                    "selected",
                    "api_requests",
                ),
                (mapping["api_requests"], mapping["api"], True, "api_requests", "api"),
                (mapping["api"], mapping["train"], True, "api", "train"),
            ]
        )
    return chain


def _row_id_chain_from_data_artifacts(data_root: Path) -> list[tuple[Path, Path, bool, str, str]]:
    chain: list[tuple[Path, Path, bool, str, str]] = []
    train = data_root / "datapool.train.jsonl"
    sampled = data_root / "datapool.train.sampled.jsonl"
    if train.exists() and sampled.exists():
        chain.append((train, sampled, True, "datapool.train", "datapool.train.sampled"))
    return chain


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate local JSONL artifacts")
    parser.add_argument("--config", default="configs/scp_stage4.yaml")
    parser.add_argument("--run-id", default=None)
    args, overrides = parser.parse_known_args(argv)

    cfg = compose_config(args.config, overrides=overrides)
    run_id = args.run_id or str(cfg.get("run", {}).get("run_id", "local_contract"))
    root_dir = Path(str(cfg.get("logging", {}).get("local", {}).get("root_dir", "artifacts/runs")))
    run_root = root_dir / run_id

    paths = _collect_jsonl_paths(run_root)
    if not paths:
        print("validate-jsonl failed: no JSONL files found", file=sys.stderr)
        return 1

    read_jsonl, validate_row_id_preservation = _resolve_data_helpers()
    validate_artifact_rows = _resolve_schema_helper()

    try:
        for path in paths:
            rows = _load_rows(path, read_jsonl)
            _validate_with_schema(path, rows, validate_artifact_rows)

        chains = []
        chains.extend(_row_id_chain_from_fixture_dir(Path("tests/fixtures")))
        chains.extend(_row_id_chain_from_run_root(run_root))
        chains.extend(_row_id_chain_from_data_artifacts(Path("artifacts/data")))

        if validate_row_id_preservation is not None:
            for base, candidate, allow_subset, base_name, candidate_name in chains:
                base_rows = _load_rows(base, read_jsonl)
                candidate_rows = _load_rows(candidate, read_jsonl)
                validate_row_id_preservation(
                    base_rows,
                    candidate_rows,
                    allow_subset=allow_subset,
                    base_name=base_name,
                    candidate_name=candidate_name,
                )

    except Exception as exc:
        print(f"validate-jsonl failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"validated_files": len(paths), "run_id": run_id}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
