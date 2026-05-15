from __future__ import annotations

import json
import errno
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scp_stage4.pipeline.prepared_data_hub import (  # noqa: E402
    package_prepared_data,
    _materialize_bundle_file,
    restore_prepared_data_from_hub,
    upload_prepared_data_bundle,
)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_parquet_rows(path: Path, rows: list[dict[str, object]]) -> None:
    parquet = pytest.importorskip("pyarrow.parquet")
    pyarrow = pytest.importorskip("pyarrow")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pyarrow.Table.from_pylist(rows)
    parquet.write_table(table, path)


def test_package_prepared_data_writes_bundle_manifest_and_config_artifacts(tmp_path: Path) -> None:
    source_dir = tmp_path / "artifacts" / "data"
    _write_parquet_rows(
        source_dir / "datapool.normalized.parquet",
        [{"id": "n-1", "source": "a"}, {"id": "n-2", "source": "b"}],
    )
    _write_parquet_rows(
        source_dir / "datapool.train.parquet",
        [{"id": "t-1", "source": "a"}],
    )
    _write_parquet_rows(
        source_dir / "datapool.eval.parquet",
        [{"id": "e-1", "source": "b"}],
    )
    _write_text(source_dir / "prepare_data_summary.json", "{}\n")

    result = package_prepared_data(
        config_path=str(ROOT / "configs" / "scp_stage4.yaml"),
        overrides=None,
        artifacts_dir=source_dir,
        output_root=tmp_path / "prepared_data_bundles",
        tag="unit-v1",
    )

    bundle_dir = Path(result["bundle_dir"])
    assert bundle_dir.exists()
    assert (bundle_dir / "effective_config.yaml").exists()
    assert (bundle_dir / "config_hash.txt").exists()
    assert (bundle_dir / "prepared_manifest.json").exists()
    assert (bundle_dir / "datapool.normalized.parquet").exists()
    assert (bundle_dir / "datapool.train.parquet").exists()
    assert (bundle_dir / "datapool.eval.parquet").exists()
    assert (bundle_dir / "prepare_data_summary.json").exists()

    manifest = json.loads((bundle_dir / "prepared_manifest.json").read_text(encoding="utf-8"))
    assert manifest["bundle_tag"] == "unit-v1"
    assert manifest["config_hash"] == result["config_hash"]
    assert sorted(entry["path"] for entry in manifest["files"]) == [
        "datapool.eval.parquet",
        "datapool.normalized.parquet",
        "datapool.train.parquet",
        "prepare_data_summary.json",
    ]
    by_path = {entry["path"]: entry for entry in manifest["files"]}
    assert by_path["datapool.train.parquet"]["format"] == "parquet"
    assert by_path["datapool.train.parquet"]["row_count"] == 1
    assert isinstance(by_path["datapool.train.parquet"]["row_id_sha256"], str)
    assert by_path["prepare_data_summary.json"]["format"] == "other"
    assert by_path["prepare_data_summary.json"]["row_count"] is None
    assert by_path["prepare_data_summary.json"]["row_id_sha256"] is None
    assert (bundle_dir / "config_hash.txt").read_text(encoding="utf-8").strip() == result[
        "config_hash"
    ]


def test_package_prepared_data_can_skip_optional_artifacts(tmp_path: Path) -> None:
    source_dir = tmp_path / "artifacts" / "data"
    _write_parquet_rows(
        source_dir / "datapool.normalized.parquet",
        [{"id": "n-1", "source": "a"}],
    )
    _write_parquet_rows(
        source_dir / "datapool.train.parquet",
        [{"id": "t-1", "source": "a"}],
    )
    _write_parquet_rows(
        source_dir / "datapool.eval.parquet",
        [{"id": "e-1", "source": "b"}],
    )
    _write_text(source_dir / "prepare_data_summary.json", "{}\n")
    _write_text(source_dir / "datapool.train.jsonl", '{"id":"optional"}\n')

    result = package_prepared_data(
        config_path=str(ROOT / "configs" / "scp_stage4.yaml"),
        overrides=None,
        artifacts_dir=source_dir,
        output_root=tmp_path / "prepared_data_bundles",
        tag="unit-required-only",
        include_optional=False,
    )

    bundle_dir = Path(result["bundle_dir"])
    assert (bundle_dir / "datapool.train.parquet").exists()
    assert not (bundle_dir / "datapool.train.jsonl").exists()
    manifest = json.loads((bundle_dir / "prepared_manifest.json").read_text(encoding="utf-8"))
    assert "datapool.train.jsonl" not in {entry["path"] for entry in manifest["files"]}


def test_upload_prepared_data_bundle_creates_tag(monkeypatch, tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_text(bundle_dir / "dummy.txt", "ok\n")

    observed: dict[str, object] = {}

    class _FakeApi:
        def __init__(self) -> None:
            observed["api"] = self
            self.create_repo_calls: list[dict[str, object]] = []
            self.upload_folder_calls: list[dict[str, object]] = []
            self.create_tag_calls: list[dict[str, object]] = []

        def create_repo(self, **kwargs):
            self.create_repo_calls.append(dict(kwargs))

        def upload_folder(self, **kwargs):
            self.upload_folder_calls.append(dict(kwargs))
            return types.SimpleNamespace(
                commit_url="https://huggingface.co/datasets/me/repo/commit/abc123",
                oid="abc123",
                pr_url=None,
            )

        def create_tag(self, **kwargs):
            self.create_tag_calls.append(dict(kwargs))

    fake_module = types.SimpleNamespace(HfApi=_FakeApi)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    result = upload_prepared_data_bundle(
        repo_id="me/repo",
        bundle_dir=bundle_dir,
        path_in_repo=None,
        revision="main",
        private=False,
        create_repo=True,
        commit_message="upload test",
        tag="v1.0.0",
        tag_message="prepared bundle",
        tag_exist_ok=True,
    )

    fake_api = observed["api"]
    assert isinstance(fake_api, _FakeApi)
    assert len(fake_api.create_repo_calls) == 1
    assert len(fake_api.upload_folder_calls) == 1
    assert len(fake_api.create_tag_calls) == 1
    tag_call = fake_api.create_tag_calls[0]
    assert tag_call["tag"] == "v1.0.0"
    assert tag_call["revision"] == "abc123"
    assert tag_call["repo_type"] == "dataset"
    assert result["tag"] == "v1.0.0"
    assert result["tagged_revision"] == "abc123"


def test_restore_prepared_data_from_hub_restores_required_files(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}
    local_download_dir = tmp_path / "prepared_data_download"
    target_subdir = "prepared/v1"
    source_bundle = local_download_dir / target_subdir

    _write_parquet_rows(
        source_bundle / "datapool.normalized.parquet",
        [{"id": "n-1", "source": "n"}],
    )
    _write_parquet_rows(
        source_bundle / "datapool.train.parquet",
        [{"id": "t-1", "source": "t"}],
    )
    _write_parquet_rows(
        source_bundle / "datapool.eval.parquet",
        [{"id": "e-1", "source": "e"}],
    )
    _write_text(source_bundle / "datapool.train.jsonl", '{"id":"legacy-jsonl"}\n')
    _write_text(source_bundle / "datapool.eval.jsonl", '{"id":"legacy-jsonl"}\n')
    _write_text(source_bundle / "datapool.normalized.jsonl", '{"id":"legacy-jsonl"}\n')
    _write_text(source_bundle / "prepare_data_summary.json", "{\"ok\":true}\n")
    _write_text(source_bundle / "effective_config.yaml", "run:\n  run_id: test\n")
    _write_text(source_bundle / "config_hash.txt", "abc\n")
    _write_text(source_bundle / "prepared_manifest.json", "{\"schema_version\":1}\n")

    def _snapshot_download(**kwargs):
        if "local_dir_use_symlinks" in kwargs:
            raise TypeError("unexpected keyword argument 'local_dir_use_symlinks'")
        observed["snapshot_kwargs"] = dict(kwargs)
        return str(source_bundle)

    fake_module = types.SimpleNamespace(snapshot_download=_snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    output_dir = tmp_path / "artifacts" / "data"
    result = restore_prepared_data_from_hub(
        repo_id="me/repo",
        path_in_repo=target_subdir,
        output_dir=output_dir,
        revision="v1",
        local_download_dir=local_download_dir,
    )

    assert result["restored_files"] == [
        "datapool.eval.parquet",
        "datapool.normalized.parquet",
        "datapool.train.parquet",
        "prepare_data_summary.json",
    ]
    assert (output_dir / "datapool.normalized.parquet").exists()
    assert (output_dir / "datapool.train.parquet").exists()
    assert (output_dir / "datapool.eval.parquet").exists()
    assert not (output_dir / "datapool.normalized.jsonl").exists()
    assert not (output_dir / "datapool.train.jsonl").exists()
    assert not (output_dir / "datapool.eval.jsonl").exists()
    assert (output_dir / "prepare_data_summary.json").exists()
    assert (output_dir / "effective_config.yaml").exists()
    assert (output_dir / "config_hash.txt").exists()
    assert (output_dir / "prepared_manifest.json").exists()

    snapshot_kwargs = observed["snapshot_kwargs"]
    assert isinstance(snapshot_kwargs, dict)
    assert snapshot_kwargs["repo_type"] == "dataset"
    assert snapshot_kwargs["revision"] == "v1"


def test_restore_prepared_data_from_hub_supports_sharded_train_eval_layout(
    monkeypatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}
    local_download_dir = tmp_path / "prepared_data_download"
    target_subdir = "prepared/prepared-2026-05-01"
    source_bundle = local_download_dir / target_subdir

    _write_parquet_rows(
        source_bundle / "train" / "part-00000.parquet",
        [{"id": "t-1", "source": "t1"}],
    )
    _write_parquet_rows(
        source_bundle / "train" / "part-00001.parquet",
        [{"id": "t-2", "source": "t2"}],
    )
    _write_parquet_rows(
        source_bundle / "eval" / "part-00000.parquet",
        [{"id": "e-1", "source": "e1"}],
    )
    _write_text(source_bundle / "manifest.json", "{\"schema_version\":1}\n")

    def _snapshot_download(**kwargs):
        if "local_dir_use_symlinks" in kwargs:
            raise TypeError("unexpected keyword argument 'local_dir_use_symlinks'")
        observed["snapshot_kwargs"] = dict(kwargs)
        return str(source_bundle)

    fake_module = types.SimpleNamespace(snapshot_download=_snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    output_dir = tmp_path / "artifacts" / "data"
    result = restore_prepared_data_from_hub(
        repo_id="alwaysgood/scp-stage4-dataset-trainable",
        path_in_repo=target_subdir,
        output_dir=output_dir,
        revision="main",
        local_download_dir=local_download_dir,
    )

    assert sorted(result["restored_files"]) == [
        "datapool.eval.parquet",
        "datapool.normalized.parquet",
        "datapool.train.parquet",
        "prepare_data_summary.json",
    ]
    assert (output_dir / "datapool.train.parquet").exists()
    assert (output_dir / "datapool.eval.parquet").exists()
    assert (output_dir / "datapool.normalized.parquet").exists()
    assert (output_dir / "prepare_data_summary.json").exists()
    assert (output_dir / "prepared_manifest.json").exists()


def test_materialize_bundle_file_falls_back_to_copy(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    target = tmp_path / "target.jsonl"
    _write_text(source, '{"id":"x"}\n')

    def _raise_cross_device(src: Path, dst: Path) -> None:
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr("scp_stage4.pipeline.prepared_data_hub.os.link", _raise_cross_device)
    mode = _materialize_bundle_file(source, target)
    assert mode == "copy"
    assert target.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_materialize_bundle_file_propagates_unexpected_oserror(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    target = tmp_path / "target.jsonl"
    _write_text(source, '{"id":"x"}\n')

    def _raise_quota(src: Path, dst: Path) -> None:
        raise OSError(errno.EDQUOT, "quota exceeded")

    monkeypatch.setattr("scp_stage4.pipeline.prepared_data_hub.os.link", _raise_quota)
    with pytest.raises(OSError):
        _materialize_bundle_file(source, target)
