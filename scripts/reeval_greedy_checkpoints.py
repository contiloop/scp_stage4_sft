#!/usr/bin/env python3
"""Restore archived run checkpoints from Hugging Face and rerun greedy OOD eval."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Iterable


ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar.xz", ".tar")
REPO_TYPES = ("model", "dataset", "space")


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _safe_extract_tar(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest_dir.resolve()
    with tarfile.open(archive_path) as handle:
        for member in handle.getmembers():
            target = (dest_dir / member.name).resolve()
            if dest_resolved not in target.parents and target != dest_resolved:
                raise RuntimeError(f"unsafe archive member path: {member.name}")
        handle.extractall(dest_dir)


def _download_from_hf(
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    subset_idx: int,
    download_dir: Path,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        from huggingface_hub.errors import RepositoryNotFoundError
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required. Run `pip install huggingface_hub` in the remote env."
        ) from exc

    subset_name = f"subset_{subset_idx:03d}"
    repo_types = [repo_type] + [kind for kind in REPO_TYPES if kind != repo_type]
    files: list[str] | None = None
    resolved_repo_type: str | None = None
    repo_errors: list[str] = []
    for candidate_repo_type in repo_types:
        try:
            files = list_repo_files(
                repo_id=repo_id,
                repo_type=candidate_repo_type,
                revision=revision,
            )
            resolved_repo_type = candidate_repo_type
            if candidate_repo_type != repo_type:
                print(
                    f"[reeval] repo_type={repo_type!r} not found; "
                    f"using repo_type={candidate_repo_type!r}",
                    file=sys.stderr,
                )
            break
        except RepositoryNotFoundError as exc:
            repo_errors.append(f"{candidate_repo_type}: {exc}")
    if files is None or resolved_repo_type is None:
        detail = "\n".join(repo_errors)
        raise RuntimeError(
            f"repo not found as any supported type for {repo_id}@{revision}. "
            "If it is private, run `huggingface-cli login` or set HF_TOKEN.\n"
            f"{detail}"
        )

    candidates = [
        path
        for path in files
        if subset_name in path and path.endswith(ARCHIVE_SUFFIXES)
    ]
    if not candidates:
        loose = [
            path
            for path in files
            if f"{subset_idx:03d}" in path and path.endswith(ARCHIVE_SUFFIXES)
        ]
        candidates = loose
    if not candidates:
        raise RuntimeError(
            f"no archive found for {subset_name} in {resolved_repo_type} repo {repo_id}@{revision}"
        )

    preferred = sorted(candidates, key=lambda path: ("/archives/" not in path, len(path), path))[0]
    local_path = hf_hub_download(
        repo_id=repo_id,
        repo_type=resolved_repo_type,
        revision=revision,
        filename=preferred,
        local_dir=download_dir,
    )
    return Path(local_path)


def _find_subset_root(extract_root: Path, subset_idx: int) -> Path:
    subset_name = f"subset_{subset_idx:03d}"
    direct = extract_root / subset_name
    if direct.exists():
        return direct
    matches = [path for path in extract_root.rglob(subset_name) if path.is_dir()]
    if not matches:
        raise RuntimeError(f"extracted archive does not contain {subset_name}")
    return matches[0]


def _checkpoint_candidates(subset_root: Path) -> Iterable[Path]:
    train_final = subset_root / "train_final"
    state_path = train_final / "checkpoint_state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            checkpoint_path = state.get("checkpoint_path")
            if isinstance(checkpoint_path, str) and checkpoint_path.strip():
                p = Path(checkpoint_path)
                if not p.is_absolute():
                    p = train_final / p
                yield p
        except Exception:
            pass
    yield train_final / "full_weight_model"
    yield train_final / "merged_model"
    yield train_final / "main_adapter"
    yield train_final


def _is_usable_checkpoint(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(
        (path / name).exists()
        for name in (
            "adapter_config.json",
            "config.json",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
    ) or any(path.glob("*.safetensors"))


def _resolve_checkpoint_path(subset_root: Path) -> Path:
    for candidate in _checkpoint_candidates(subset_root):
        if _is_usable_checkpoint(candidate):
            return candidate
    raise RuntimeError(f"no usable checkpoint found under {subset_root / 'train_final'}")


def _write_latest_pointer(
    *,
    run_root: Path,
    run_id: str,
    subset_idx: int,
    checkpoint_path: Path,
    source_archive: Path,
) -> None:
    payload = {
        "status": "ok",
        "run_id": run_id,
        "subset_idx": subset_idx,
        "checkpoint_path": str(checkpoint_path),
        "source_archive": str(source_archive),
    }
    latest_path = run_root / "checkpoints" / "latest.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="alwaysgood/scp-stage4-run-main-001")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--revision", default="main")
    parser.add_argument("--config", default="configs/scp_stage4_real_1gpu_greedy_eval.yaml")
    parser.add_argument("--run-id", default="greedy_reeval_main_001")
    parser.add_argument("--checkpoint-indices", nargs="+", type=int, default=[17, 19, 31, 32])
    parser.add_argument("--download-dir", default="artifacts/hf_downloads/scp-stage4-run-main-001")
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--skip-download", action="store_true")
    args, overrides = parser.parse_known_args(argv)

    run_root = Path(args.run_root) if args.run_root else Path("artifacts/runs") / args.run_id
    download_dir = Path(args.download_dir)

    for subset_idx in args.checkpoint_indices:
        subset_name = f"subset_{subset_idx:03d}"
        extract_root = download_dir / "extracted" / subset_name
        if not args.skip_download:
            archive_path = _download_from_hf(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                revision=args.revision,
                subset_idx=subset_idx,
                download_dir=download_dir,
            )
            if extract_root.exists():
                shutil.rmtree(extract_root)
            _safe_extract_tar(archive_path, extract_root)
        else:
            archives = sorted((download_dir).rglob(f"*{subset_name}*"))
            archives = [p for p in archives if p.name.endswith(ARCHIVE_SUFFIXES)]
            if not archives:
                raise RuntimeError(f"--skip-download set but no local archive found for {subset_name}")
            archive_path = archives[0]
            if not extract_root.exists():
                _safe_extract_tar(archive_path, extract_root)

        restored_subset = _find_subset_root(extract_root, subset_idx)
        target_subset = run_root / "subsets" / subset_name
        if target_subset.exists():
            shutil.rmtree(target_subset)
        target_subset.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(restored_subset, target_subset)

        checkpoint_path = _resolve_checkpoint_path(target_subset)
        _write_latest_pointer(
            run_root=run_root,
            run_id=args.run_id,
            subset_idx=subset_idx,
            checkpoint_path=checkpoint_path,
            source_archive=archive_path,
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
