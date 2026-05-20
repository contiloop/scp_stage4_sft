#!/usr/bin/env python3
"""Replay saved API corrections through cumulative update-base + greedy eval."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from reeval_greedy_checkpoints import (
    HfPayload,
    _download_from_hf,
    _find_subset_root,
    _is_archive_path,
    _safe_extract_archive,
)


REQUIRED_REPLAY_FILES = ("api.jsonl", "clean_base.json")


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _subset_name(subset_idx: int) -> str:
    return f"subset_{subset_idx:03d}"


def _restore_subset_artifacts(
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    subset_idx: int,
    download_dir: Path,
    run_root: Path,
    skip_download: bool,
    keep_download_payload: bool,
) -> Path:
    subset_name = _subset_name(subset_idx)
    extract_root = download_dir / "replay_extracted" / subset_name

    if not skip_download:
        payload = _download_from_hf(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            subset_idx=subset_idx,
            download_dir=download_dir,
        )
        if payload.kind == "archive":
            if extract_root.exists():
                shutil.rmtree(extract_root)
            _safe_extract_archive(payload.path, extract_root)
            payload_root = extract_root
        else:
            payload_root = payload.path
    else:
        archives = sorted(download_dir.rglob(f"*{subset_name}*"))
        archives = [path for path in archives if _is_archive_path(path.name)]
        if archives:
            archive_path = archives[0]
            if not extract_root.exists():
                _safe_extract_archive(archive_path, extract_root)
            payload = HfPayload(path=archive_path, kind="archive", source=str(archive_path))
            payload_root = extract_root
        else:
            payload_root = download_dir / "snapshots" / subset_name
            payload = HfPayload(path=payload_root, kind="tree", source=str(payload_root))

    source_subset = _find_subset_root(payload_root, subset_idx)
    target_subset = run_root / "subsets" / subset_name
    if target_subset.exists():
        shutil.rmtree(target_subset)
    target_subset.parent.mkdir(parents=True, exist_ok=True)
    target_subset.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_REPLAY_FILES:
        shutil.copy2(source_subset / name, target_subset / name)
    for optional_name in ("selected.jsonl", "api_requests.jsonl"):
        optional_path = source_subset / optional_name
        if optional_path.exists():
            shutil.copy2(optional_path, target_subset / optional_name)

    missing = [name for name in REQUIRED_REPLAY_FILES if not (target_subset / name).exists()]
    if missing:
        raise RuntimeError(
            f"{subset_name} missing replay artifacts after restore: {', '.join(missing)} "
            f"(source={payload.source})"
        )

    if not keep_download_payload and not skip_download:
        if payload.kind == "archive" and payload.path.exists():
            payload.path.unlink()
        if extract_root.exists():
            shutil.rmtree(extract_root)
        snapshot_root = download_dir / "snapshots" / subset_name
        if snapshot_root.exists():
            shutil.rmtree(snapshot_root)
    return target_subset


def _write_replay_manifest(
    *,
    run_root: Path,
    repo_id: str,
    revision: str,
    subset_indices: list[int],
    clean_run: bool,
) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "started",
        "repo_id": repo_id,
        "revision": revision,
        "subset_indices": subset_indices,
        "clean_run": clean_run,
    }
    (run_root / "replay_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _mark_replay_complete(run_root: Path) -> None:
    path = run_root / "replay_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "ok"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _latest_checkpoint_path(run_root: Path) -> Path:
    return run_root / "checkpoints" / "latest.json"


def _assert_no_latest_checkpoint(run_root: Path) -> None:
    latest_path = _latest_checkpoint_path(run_root)
    if latest_path.exists():
        raise RuntimeError(
            "replay must start subset_000 from the configured base model, but "
            f"{latest_path} already exists. Use --clean-run or a new --run-id."
        )


def _assert_latest_checkpoint_exists(run_root: Path, *, subset_idx: int) -> None:
    latest_path = _latest_checkpoint_path(run_root)
    if not latest_path.exists():
        raise RuntimeError(
            f"subset_{subset_idx:03d} replay would fall back to model.name because "
            f"{latest_path} is missing"
        )
    state = json.loads(latest_path.read_text(encoding="utf-8"))
    checkpoint = state.get("checkpoint_path")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise RuntimeError(f"{latest_path} missing checkpoint_path")
    if not Path(checkpoint).exists():
        raise RuntimeError(
            f"subset_{subset_idx:03d} replay would fall back to model.name because "
            f"latest checkpoint path does not exist: {checkpoint}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="alwaysgood/scp-stage4-run-main-001")
    parser.add_argument("--repo-type", default="dataset", choices=["model", "dataset", "space"])
    parser.add_argument("--revision", default="main")
    parser.add_argument("--config", default="configs/scp_stage4_real_1gpu_greedy_eval.yaml")
    parser.add_argument("--run-id", default="replay_main_001_greedy")
    parser.add_argument("--start-subset", type=int, default=0)
    parser.add_argument("--end-subset", type=int, default=32)
    parser.add_argument("--subset-indices", nargs="+", type=int, default=None)
    parser.add_argument("--download-dir", default="artifacts/hf_downloads/scp-stage4-run-main-001")
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--clean-run", action="store_true")
    parser.add_argument("--clean-download-cache", action="store_true")
    parser.add_argument("--keep-download-payload", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    args, overrides = parser.parse_known_args(argv)

    subset_indices = (
        [int(idx) for idx in args.subset_indices]
        if args.subset_indices is not None
        else list(range(int(args.start_subset), int(args.end_subset) + 1))
    )
    if not subset_indices:
        raise SystemExit("no subset indices requested")

    run_root = Path(args.run_root) if args.run_root else Path("artifacts/runs") / args.run_id
    download_dir = Path(args.download_dir)

    if args.clean_run and run_root.exists():
        shutil.rmtree(run_root)
    if args.clean_download_cache and download_dir.exists():
        shutil.rmtree(download_dir)

    _write_replay_manifest(
        run_root=run_root,
        repo_id=args.repo_id,
        revision=args.revision,
        subset_indices=subset_indices,
        clean_run=bool(args.clean_run),
    )

    for subset_idx in subset_indices:
        if subset_idx == 0:
            _assert_no_latest_checkpoint(run_root)
            checkpoint_guard_overrides = []
        else:
            _assert_latest_checkpoint_exists(run_root, subset_idx=subset_idx)
            checkpoint_guard_overrides = ["training.base_update.requires_base_checkpoint=true"]

        subset_name = _subset_name(subset_idx)
        print(f"[replay] restoring {subset_name}", file=sys.stderr)
        _restore_subset_artifacts(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            subset_idx=subset_idx,
            download_dir=download_dir,
            run_root=run_root,
            skip_download=bool(args.skip_download),
            keep_download_payload=bool(args.keep_download_payload),
        )

        _run(
            [
                sys.executable,
                "-m",
                "scp_stage4.pipeline.step_subset",
                "update-base",
                "--config",
                args.config,
                "--run-id",
                args.run_id,
                "--subset-idx",
                str(subset_idx),
                *checkpoint_guard_overrides,
                *overrides,
            ]
        )
        _run(
            [
                sys.executable,
                "-m",
                "scp_stage4.pipeline.step_subset",
                "eval-ood",
                "--config",
                args.config,
                "--run-id",
                args.run_id,
                "--subset-idx",
                str(subset_idx),
                *overrides,
            ]
        )

    _mark_replay_complete(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
