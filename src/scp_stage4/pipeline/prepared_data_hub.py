"""Package, upload, and restore prepared datapool artifacts via Hugging Face Hub."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Iterable, Mapping

from scp_stage4.artifacts import compute_config_hash, persist_effective_config_artifacts
from scp_stage4.config.loader import compose_config
from scp_stage4.config.validator import validate_config


class PreparedDataHubError(RuntimeError):
    """Raised when prepared-data packaging/upload/download cannot complete."""


_REQUIRED_FILES = (
    "datapool.normalized.parquet",
    "datapool.train.parquet",
    "datapool.eval.parquet",
    "prepare_data_summary.json",
)
_OPTIONAL_PACK_FILES = (
    "datapool.normalized.jsonl",
    "datapool.train.jsonl",
    "datapool.eval.jsonl",
    "datapool.train.sampled.parquet",
    "datapool.train.sampled.jsonl",
    "ood_test.jsonl",
)
_OPTIONAL_RESTORE_FILES = (
    "datapool.train.sampled.parquet",
    "ood_test.jsonl",
)

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl_row_ids(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", newline="\n") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise PreparedDataHubError(
                    f"Invalid JSONL in prepared artifact {path}:{line_no}: {exc}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise PreparedDataHubError(
                    f"Invalid JSONL object in prepared artifact {path}:{line_no}"
                )
            row_id = payload.get("id")
            if row_id is None:
                raise PreparedDataHubError(
                    f"Missing id in prepared artifact {path}:{line_no}"
                )
            yield str(row_id)


def _iter_parquet_row_ids(path: Path) -> Iterable[str]:
    if pa is None or pq is None:
        raise PreparedDataHubError(
            f"pyarrow is required to compute row_id_hash for parquet artifact: {path}"
        )
    parquet_file = pq.ParquetFile(str(path))
    for record_batch in parquet_file.iter_batches(batch_size=4096, columns=["id"]):
        table = pa.Table.from_batches([record_batch])
        for row in table.to_pylist():
            if not isinstance(row, Mapping):
                continue
            row_id = row.get("id")
            if row_id is None:
                raise PreparedDataHubError(f"Missing id in prepared parquet artifact: {path}")
            yield str(row_id)


def _dataset_file_stats(path: Path) -> tuple[str, int | None, str | None]:
    suffix = path.suffix.lower()
    if suffix not in {".jsonl", ".parquet"}:
        return "other", None, None

    if suffix == ".jsonl":
        row_ids = _iter_jsonl_row_ids(path)
        fmt = "jsonl"
    else:
        row_ids = _iter_parquet_row_ids(path)
        fmt = "parquet"

    row_count = 0
    digest = hashlib.sha256()
    for row_id in row_ids:
        digest.update(row_id.encode("utf-8"))
        digest.update(b"\n")
        row_count += 1
    return fmt, row_count, digest.hexdigest()


def _copy_required_artifacts(
    *,
    source_dir: Path,
    target_dir: Path,
    include_optional: bool = True,
) -> list[str]:
    copied: list[str] = []
    missing: list[str] = []

    for name in _REQUIRED_FILES:
        source_path = source_dir / name
        if not source_path.exists():
            missing.append(name)
            continue
        _materialize_bundle_file(source_path, target_dir / name)
        copied.append(name)

    if missing:
        raise PreparedDataHubError(
            "Missing required prepared-data artifacts: " + ", ".join(sorted(missing))
        )

    if include_optional:
        for name in _OPTIONAL_PACK_FILES:
            source_path = source_dir / name
            if not source_path.exists():
                continue
            _materialize_bundle_file(source_path, target_dir / name)
            copied.append(name)

    return copied


def _materialize_bundle_file(source: Path, target: Path) -> str:
    """
    Materialize bundle file with minimal extra disk usage.

    Returns "hardlink" or "copy".
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
        return "hardlink"
    except OSError as exc:
        if exc.errno in {
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            errno.ENOTSUP,
            errno.ENOSYS,
            errno.EMLINK,
        }:
            shutil.copy2(source, target)
            return "copy"
        raise


def package_prepared_data(
    *,
    config_path: str,
    overrides: list[str] | None,
    artifacts_dir: str | Path,
    output_root: str | Path,
    tag: str | None = None,
    force: bool = False,
    include_optional: bool = True,
) -> dict[str, Any]:
    cfg = compose_config(config_path, overrides=overrides)
    validate_config(cfg)

    config_hash = compute_config_hash(cfg)
    bundle_tag = tag.strip() if isinstance(tag, str) and tag.strip() else config_hash[:12]

    source_dir = Path(artifacts_dir)
    if not source_dir.exists():
        raise PreparedDataHubError(f"artifacts dir not found: {source_dir}")

    bundle_dir = Path(output_root) / bundle_tag
    if bundle_dir.exists():
        if not force:
            raise PreparedDataHubError(
                f"bundle already exists: {bundle_dir} (use --force to replace)"
            )
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    copied_files = _copy_required_artifacts(
        source_dir=source_dir,
        target_dir=bundle_dir,
        include_optional=include_optional,
    )
    persisted = persist_effective_config_artifacts(
        run_dir=bundle_dir,
        effective_config=cfg,
        write_effective_config=True,
        write_config_hash=True,
    )
    if str(persisted["config_hash"]) != config_hash:
        raise PreparedDataHubError("config_hash mismatch while packaging prepared data")

    file_entries: list[dict[str, Any]] = []
    for filename in sorted(copied_files):
        path = bundle_dir / filename
        file_format, row_count, row_id_hash = _dataset_file_stats(path)
        file_entries.append(
            {
                "path": filename,
                "format": file_format,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "row_count": row_count,
                "row_id_sha256": row_id_hash,
            }
        )

    manifest = {
        "schema_version": 2,
        "bundle_tag": bundle_tag,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
        "config_overrides": list(overrides or []),
        "config_hash": config_hash,
        "source_artifacts_dir": str(source_dir),
        "files": file_entries,
    }
    manifest_path = bundle_dir / "prepared_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "bundle_dir": str(bundle_dir),
        "bundle_tag": bundle_tag,
        "config_hash": config_hash,
        "manifest_path": str(manifest_path),
        "copied_files": sorted(copied_files),
    }


def _extract_commit_fields(commit_info: Any) -> dict[str, Any]:
    return {
        "commit_url": getattr(commit_info, "commit_url", None),
        "oid": getattr(commit_info, "oid", None),
        "pr_url": getattr(commit_info, "pr_url", None),
    }


def upload_prepared_data_bundle(
    *,
    repo_id: str,
    bundle_dir: str | Path,
    path_in_repo: str | None,
    revision: str,
    private: bool,
    create_repo: bool,
    commit_message: str | None,
    tag: str | None,
    tag_message: str | None,
    tag_exist_ok: bool,
) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ModuleNotFoundError as exc:
        raise PreparedDataHubError(
            "upload requires huggingface_hub; install it and run `huggingface-cli login` first"
        ) from exc

    bundle_path = Path(bundle_dir)
    if not bundle_path.exists():
        raise PreparedDataHubError(f"bundle dir not found: {bundle_path}")

    target_path = path_in_repo.strip() if isinstance(path_in_repo, str) and path_in_repo.strip() else f"prepared/{bundle_path.name}"
    message = commit_message or f"Add prepared-data bundle {bundle_path.name}"

    api = HfApi()
    if create_repo:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(bundle_path),
        path_in_repo=target_path,
        revision=revision,
        commit_message=message,
    )
    tag_name = tag.strip() if isinstance(tag, str) and tag.strip() else None
    tagged_revision: str | None = None
    if tag_name is not None:
        tagged_revision = getattr(commit_info, "oid", None) or revision
        api.create_tag(
            repo_id=repo_id,
            repo_type="dataset",
            tag=tag_name,
            tag_message=tag_message,
            revision=tagged_revision,
            exist_ok=bool(tag_exist_ok),
        )
    return {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "bundle_dir": str(bundle_path),
        "path_in_repo": target_path,
        "revision": revision,
        "tag": tag_name,
        "tagged_revision": tagged_revision,
        **_extract_commit_fields(commit_info),
    }


def _copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        shutil.copy2(source, target)


def _write_merged_parquet_from_parts(parts: list[Path], output_path: Path) -> None:
    if pa is None or pq is None:
        raise PreparedDataHubError(
            f"pyarrow is required to restore sharded prepared bundle into {output_path.name}"
        )
    if not parts:
        raise PreparedDataHubError(f"no parquet parts found for {output_path.name}")

    tables = []
    for part in parts:
        tables.append(pq.read_table(part))
    merged = pa.concat_tables(tables, promote_options="default")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(merged, output_path)


def _try_restore_sharded_layout(*, bundle_dir: Path, output_path: Path) -> list[str] | None:
    """Restore from alternate bundle layout:
    prepared/<tag>/{train,eval}/part-*.parquet + manifest.json
    """
    train_dir = bundle_dir / "train"
    eval_dir = bundle_dir / "eval"
    if not train_dir.exists() or not eval_dir.exists():
        return None

    train_parts = sorted(train_dir.glob("*.parquet"))
    eval_parts = sorted(eval_dir.glob("*.parquet"))
    if not train_parts or not eval_parts:
        return None

    _write_merged_parquet_from_parts(train_parts, output_path / "datapool.train.parquet")
    _write_merged_parquet_from_parts(eval_parts, output_path / "datapool.eval.parquet")
    # Fallback rule: normalized is unavailable in this external layout, so restore as train-equivalent.
    shutil.copy2(output_path / "datapool.train.parquet", output_path / "datapool.normalized.parquet")

    manifest_json = bundle_dir / "manifest.json"
    summary = {
        "restored_from_layout": "sharded_train_eval_parts",
        "source_bundle_dir": str(bundle_dir),
        "train_parts": len(train_parts),
        "eval_parts": len(eval_parts),
        "manifest_json_present": manifest_json.exists(),
    }
    (output_path / "prepare_data_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if manifest_json.exists():
        _copy_if_exists(manifest_json, output_path / "prepared_manifest.json")
    return [
        "datapool.eval.parquet",
        "datapool.normalized.parquet",
        "datapool.train.parquet",
        "prepare_data_summary.json",
    ]


def restore_prepared_data_from_hub(
    *,
    repo_id: str,
    path_in_repo: str,
    output_dir: str | Path,
    revision: str,
    local_download_dir: str | Path,
) -> dict[str, Any]:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ModuleNotFoundError as exc:
        raise PreparedDataHubError(
            "download requires huggingface_hub; install it and run `huggingface-cli login` first"
        ) from exc

    target_subdir = path_in_repo.strip().strip("/")
    if not target_subdir:
        raise PreparedDataHubError("path_in_repo is required for download/restore")

    local_download_path = Path(local_download_dir)
    local_download_path.mkdir(parents=True, exist_ok=True)

    _snapshot_download_compat(
        snapshot_download=snapshot_download,
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[f"{target_subdir}/*", f"{target_subdir}/**"],
        local_dir=str(local_download_path),
    )

    bundle_dir = local_download_path / target_subdir
    if not bundle_dir.exists():
        raise PreparedDataHubError(
            f"downloaded bundle path not found: {bundle_dir} (repo/path_in_repo mismatch)"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    missing_required: list[str] = []
    restored_files: list[str] = []
    for name in _REQUIRED_FILES:
        source_file = bundle_dir / name
        if not source_file.exists():
            missing_required.append(name)
            continue
        _copy_if_exists(source_file, output_path / name)
        restored_files.append(name)
    if missing_required:
        restored_from_sharded = _try_restore_sharded_layout(
            bundle_dir=bundle_dir,
            output_path=output_path,
        )
        if restored_from_sharded is None:
            raise PreparedDataHubError(
                "Downloaded bundle is missing required files: "
                + ", ".join(sorted(missing_required))
            )
        restored_files = restored_from_sharded

    for name in _OPTIONAL_RESTORE_FILES:
        source_file = bundle_dir / name
        if source_file.exists():
            _copy_if_exists(source_file, output_path / name)
            restored_files.append(name)

    _copy_if_exists(bundle_dir / "effective_config.yaml", output_path / "effective_config.yaml")
    _copy_if_exists(bundle_dir / "config_hash.txt", output_path / "config_hash.txt")
    _copy_if_exists(bundle_dir / "prepared_manifest.json", output_path / "prepared_manifest.json")

    return {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "path_in_repo": target_subdir,
        "revision": revision,
        "bundle_dir": str(bundle_dir),
        "output_dir": str(output_path),
        "restored_files": sorted(restored_files),
    }


def _snapshot_download_compat(snapshot_download: Callable[..., Any], **kwargs: Any) -> Any:
    """Call snapshot_download across huggingface_hub versions."""
    try:
        return snapshot_download(local_dir_use_symlinks=False, **kwargs)
    except TypeError:
        return snapshot_download(**kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepared-data packaging/upload/restore helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack_parser = subparsers.add_parser("pack", help="Package artifacts/data into a versioned bundle")
    pack_parser.add_argument("--config", default="configs/scp_stage4_real.yaml")
    pack_parser.add_argument("--artifacts-dir", default="artifacts/data")
    pack_parser.add_argument("--output-root", default="artifacts/prepared_data_bundles")
    pack_parser.add_argument("--tag", default=None)
    pack_parser.add_argument("--force", action="store_true")
    pack_parser.add_argument(
        "--no-optional",
        action="store_true",
        help="Only package required parquet/summary artifacts; skip JSONL/sample/OOD extras.",
    )

    upload_parser = subparsers.add_parser("upload", help="Upload one prepared-data bundle to HF dataset repo")
    upload_parser.add_argument("--repo-id", required=True)
    upload_parser.add_argument("--bundle-dir", required=True)
    upload_parser.add_argument("--path-in-repo", default=None)
    upload_parser.add_argument("--revision", default="main")
    upload_parser.add_argument("--private", action="store_true")
    upload_parser.add_argument("--no-create-repo", action="store_true")
    upload_parser.add_argument("--commit-message", default=None)
    upload_parser.add_argument("--tag", default=None)
    upload_parser.add_argument("--tag-message", default=None)
    upload_parser.add_argument("--tag-exist-ok", action="store_true")

    download_parser = subparsers.add_parser(
        "download", help="Download a prepared-data bundle from HF and restore artifacts/data"
    )
    download_parser.add_argument("--repo-id", required=True)
    download_parser.add_argument("--path-in-repo", required=True)
    download_parser.add_argument("--output-dir", default="artifacts/data")
    download_parser.add_argument("--revision", default="main")
    download_parser.add_argument("--local-download-dir", default="artifacts/prepared_data_download")

    args, overrides = parser.parse_known_args(argv)

    if args.command == "pack":
        result = package_prepared_data(
            config_path=args.config,
            overrides=overrides,
            artifacts_dir=args.artifacts_dir,
            output_root=args.output_root,
            tag=args.tag,
            force=bool(args.force),
            include_optional=not bool(args.no_optional),
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0

    if overrides:
        raise PreparedDataHubError(
            f"unexpected extra arguments for '{args.command}': {' '.join(overrides)}"
        )

    if args.command == "upload":
        result = upload_prepared_data_bundle(
            repo_id=args.repo_id,
            bundle_dir=args.bundle_dir,
            path_in_repo=args.path_in_repo,
            revision=args.revision,
            private=bool(args.private),
            create_repo=not bool(args.no_create_repo),
            commit_message=args.commit_message,
            tag=args.tag,
            tag_message=args.tag_message,
            tag_exist_ok=bool(args.tag_exist_ok),
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0

    if args.command == "download":
        result = restore_prepared_data_from_hub(
            repo_id=args.repo_id,
            path_in_repo=args.path_in_repo,
            output_dir=args.output_dir,
            revision=args.revision,
            local_download_dir=args.local_download_dir,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0

    raise PreparedDataHubError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreparedDataHubError as exc:
        import sys

        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
