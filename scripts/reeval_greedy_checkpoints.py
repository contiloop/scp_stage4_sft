#!/usr/bin/env python3
"""Restore archived run checkpoints from Hugging Face and rerun greedy OOD eval."""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Iterable


ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar.xz", ".tar.zst", ".tar", ".zip")
REPO_TYPES = ("model", "dataset", "space")


@dataclass(frozen=True)
class HfPayload:
    path: Path
    kind: str
    source: str


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _safe_extract_tar(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest_dir.resolve()
    if archive_path.name.endswith(".tar.zst"):
        _safe_extract_tar_zst(archive_path, dest_dir, dest_resolved)
        return
    with tarfile.open(archive_path) as handle:
        for member in handle.getmembers():
            target = (dest_dir / member.name).resolve()
            if dest_resolved not in target.parents and target != dest_resolved:
                raise RuntimeError(f"unsafe archive member path: {member.name}")
        handle.extractall(dest_dir)


def _safe_extract_tar_zst(archive_path: Path, dest_dir: Path, dest_resolved: Path) -> None:
    list_result = subprocess.run(
        ["tar", "--zstd", "-tf", str(archive_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if list_result.returncode != 0:
        detail = (list_result.stderr or list_result.stdout or "").strip()
        raise RuntimeError(f"failed to list tar.zst archive {archive_path}: {detail}")
    for raw_name in list_result.stdout.splitlines():
        name = raw_name.strip()
        if not name:
            continue
        target = (dest_dir / name).resolve()
        if dest_resolved not in target.parents and target != dest_resolved:
            raise RuntimeError(f"unsafe archive member path: {name}")
    extract_result = subprocess.run(
        ["tar", "--zstd", "-xf", str(archive_path), "-C", str(dest_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if extract_result.returncode != 0:
        detail = (extract_result.stderr or extract_result.stdout or "").strip()
        raise RuntimeError(f"failed to extract tar.zst archive {archive_path}: {detail}")


def _safe_extract_zip(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest_dir.resolve()
    with zipfile.ZipFile(archive_path) as handle:
        for member in handle.infolist():
            target = (dest_dir / member.filename).resolve()
            if dest_resolved not in target.parents and target != dest_resolved:
                raise RuntimeError(f"unsafe archive member path: {member.filename}")
        handle.extractall(dest_dir)


def _safe_extract_archive(archive_path: Path, dest_dir: Path) -> None:
    if archive_path.name.endswith(".zip"):
        _safe_extract_zip(archive_path, dest_dir)
        return
    _safe_extract_tar(archive_path, dest_dir)


def _is_archive_path(path: str) -> bool:
    return path.endswith(ARCHIVE_SUFFIXES)


def _numeric_tokens(path: str) -> set[int]:
    tokens: set[int] = set()
    for raw in re.findall(r"\d+", path):
        try:
            tokens.add(int(raw))
        except ValueError:
            pass
    return tokens


def _subset_patterns(subset_idx: int) -> tuple[str, ...]:
    subset_padded = f"{subset_idx:03d}"
    return (
        f"subset_{subset_padded}",
        f"subset-{subset_padded}",
        f"subset_{subset_idx}",
        f"subset-{subset_idx}",
        f"checkpoint_{subset_padded}",
        f"checkpoint-{subset_padded}",
        f"checkpoint_{subset_idx}",
        f"checkpoint-{subset_idx}",
        f"ckpt_{subset_padded}",
        f"ckpt-{subset_padded}",
        f"ckpt_{subset_idx}",
        f"ckpt-{subset_idx}",
    )


def _archive_matches_subset(path: str, subset_idx: int) -> bool:
    subset_patterns = _subset_patterns(subset_idx)
    lowered = path.lower()
    if any(pattern in lowered for pattern in subset_patterns):
        return True
    return subset_idx in _numeric_tokens(path)


def _tree_file_matches_subset(path: str, subset_idx: int) -> bool:
    lowered = path.lower()
    if any(pattern in lowered for pattern in _subset_patterns(subset_idx)):
        return True
    tokens = _numeric_tokens(path)
    if subset_idx not in tokens:
        return False
    return any(
        keyword in lowered
        for keyword in ("subset", "checkpoint", "ckpt", "train_final", "adapter")
    )


def _prefix_for_subset_tree(paths: list[str], subset_idx: int) -> str | None:
    for path in sorted(paths, key=lambda item: (len(item), item)):
        parts = PurePosixPath(path).parts
        for idx, part in enumerate(parts):
            lowered = part.lower()
            if any(pattern in lowered for pattern in _subset_patterns(subset_idx)):
                return "/".join(parts[: idx + 1])

    parent_dirs = [str(PurePosixPath(path).parent) for path in paths]
    if not parent_dirs:
        return None
    common = posixpath.commonpath(parent_dirs)
    return None if common in {"", "."} else common


def _download_from_hf(
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    subset_idx: int,
    download_dir: Path,
) -> HfPayload:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
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

    archive_files = [path for path in files if _is_archive_path(path)]
    candidates = [
        path
        for path in archive_files
        if _archive_matches_subset(path, subset_idx)
    ]
    if candidates:
        preferred = sorted(
            candidates,
            key=lambda path: ("/archive" not in path.lower(), len(path), path),
        )[0]
        print(
            f"[reeval] using archive for {subset_name}: {preferred}",
            file=sys.stderr,
        )
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type=resolved_repo_type,
            revision=revision,
            filename=preferred,
            local_dir=download_dir,
        )
        return HfPayload(path=Path(local_path), kind="archive", source=preferred)

    tree_files = [
        path
        for path in files
        if not _is_archive_path(path) and _tree_file_matches_subset(path, subset_idx)
    ]
    if tree_files:
        prefix = _prefix_for_subset_tree(tree_files, subset_idx)
        allow_patterns = [f"{prefix}/**"] if prefix else sorted(tree_files)
        local_dir = download_dir / "snapshots" / subset_name
        if local_dir.exists():
            shutil.rmtree(local_dir)
        print(
            f"[reeval] no archive for {subset_name}; downloading file tree "
            f"prefix={prefix or '(explicit files)'}",
            file=sys.stderr,
        )
        snapshot_download(
            repo_id=repo_id,
            repo_type=resolved_repo_type,
            revision=revision,
            allow_patterns=allow_patterns,
            local_dir=local_dir,
        )
        payload_path = local_dir / prefix if prefix else local_dir
        return HfPayload(
            path=payload_path,
            kind="tree",
            source=prefix or ",".join(sorted(tree_files)),
        )

    archive_preview = "\n".join(f"- {path}" for path in sorted(archive_files)[:50])
    if len(archive_files) > 50:
        archive_preview += f"\n... and {len(archive_files) - 50} more"
    if not archive_preview:
        archive_preview = "(no .tar/.tgz/.zip archive files found)"

    relevant = [
        path
        for path in files
        if any(token in path.lower() for token in ("subset", "checkpoint", "ckpt", "adapter", "train_final"))
        or subset_idx in _numeric_tokens(path)
    ]
    relevant_preview = "\n".join(f"- {path}" for path in sorted(relevant)[:80])
    if len(relevant) > 80:
        relevant_preview += f"\n... and {len(relevant) - 80} more"
    if not relevant_preview:
        relevant_preview = "(no subset/checkpoint-looking files found)"

    raise RuntimeError(
        f"no archive or checkpoint file tree found for {subset_name} in "
        f"{resolved_repo_type} repo {repo_id}@{revision}.\n"
        f"Archive candidates:\n{archive_preview}\n"
        f"Relevant repo files:\n{relevant_preview}"
    )


def _find_subset_root(extract_root: Path, subset_idx: int) -> Path:
    subset_name = f"subset_{subset_idx:03d}"
    candidates: list[Path] = []
    if extract_root.name == subset_name and extract_root.is_dir():
        candidates.append(extract_root)
    direct = extract_root / subset_name
    if direct.exists():
        candidates.append(direct)
    candidates.extend(
        path for path in extract_root.rglob("*") if path.is_dir() and path.name == subset_name
    )
    if not candidates:
        raise RuntimeError(f"extracted archive does not contain {subset_name}")
    unique = sorted(set(candidates), key=lambda path: (len(str(path)), str(path)))
    for candidate in unique:
        if (candidate / "api.jsonl").exists() and (candidate / "clean_base.json").exists():
            return candidate
    return unique[0]


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


def _copy_tree_payload_to_subset(
    *,
    payload_root: Path,
    target_subset: Path,
    subset_idx: int,
) -> None:
    if target_subset.exists():
        shutil.rmtree(target_subset)
    target_subset.parent.mkdir(parents=True, exist_ok=True)

    try:
        restored_subset = _find_subset_root(payload_root, subset_idx)
        shutil.copytree(restored_subset, target_subset)
        return
    except RuntimeError:
        pass

    if (payload_root / "train_final").exists():
        shutil.copytree(payload_root, target_subset)
        return

    if _is_usable_checkpoint(payload_root):
        checkpoint_dir = target_subset / "train_final" / "main_adapter"
        checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(payload_root, checkpoint_dir)
        return

    usable_dirs = [path for path in payload_root.rglob("*") if path.is_dir() and _is_usable_checkpoint(path)]
    if usable_dirs:
        checkpoint_dir = target_subset / "train_final" / "main_adapter"
        checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(sorted(usable_dirs, key=lambda path: (len(str(path)), str(path)))[0], checkpoint_dir)
        return

    raise RuntimeError(f"downloaded file tree has no usable checkpoint: {payload_root}")


def _write_latest_pointer(
    *,
    run_root: Path,
    run_id: str,
    subset_idx: int,
    checkpoint_path: Path,
    source_payload: str,
) -> None:
    payload = {
        "status": "ok",
        "run_id": run_id,
        "subset_idx": subset_idx,
        "checkpoint_path": str(checkpoint_path),
        "source_payload": source_payload,
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
            payload = _download_from_hf(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                revision=args.revision,
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
            archives = sorted((download_dir).rglob(f"*{subset_name}*"))
            archives = [p for p in archives if _is_archive_path(p.name)]
            if not archives:
                raise RuntimeError(f"--skip-download set but no local archive found for {subset_name}")
            archive_path = archives[0]
            if not extract_root.exists():
                _safe_extract_archive(archive_path, extract_root)
            payload = HfPayload(path=archive_path, kind="archive", source=str(archive_path))
            payload_root = extract_root

        target_subset = run_root / "subsets" / subset_name
        _copy_tree_payload_to_subset(
            payload_root=payload_root,
            target_subset=target_subset,
            subset_idx=subset_idx,
        )

        checkpoint_path = _resolve_checkpoint_path(target_subset)
        _write_latest_pointer(
            run_root=run_root,
            run_id=args.run_id,
            subset_idx=subset_idx,
            checkpoint_path=checkpoint_path,
            source_payload=payload.source,
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
