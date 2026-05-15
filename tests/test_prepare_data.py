from __future__ import annotations

import importlib.util
import errno
import json
import os
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scp_stage4.data import read_jsonl  # noqa: E402
from scp_stage4.pipeline import prepare_data as prepare_data_module  # noqa: E402
from scp_stage4.pipeline.prepare_data import run_prepare_data  # noqa: E402
from scp_stage4.schema import validate_artifact_rows  # noqa: E402


def test_prepare_data_writes_expected_artifacts(tmp_path: Path) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd()
    try:
        # Ensure fixture lookup falls back deterministically.
        (workdir / "tests" / "fixtures").mkdir(parents=True, exist_ok=True)
        os_config = str(ROOT / "configs" / "scp_stage4.yaml")

        # Use fixed_size strategy via overrides for deterministic sample count.
        os.chdir(workdir)
        summary = run_prepare_data(
            config_path=os_config,
            overrides=[
                "pipeline.subset.strategy=fixed_size",
                "pipeline.subset.fixed_size=32",
                "data.sampling.strategy=first_n",
                "data.subset_size=32",
            ],
        )

        out_dir = workdir / "artifacts" / "data"
        expected = [
            out_dir / "datapool.normalized.jsonl",
            out_dir / "datapool.train.jsonl",
            out_dir / "datapool.eval.jsonl",
            out_dir / "datapool.train.sampled.jsonl",
            out_dir / "ood_test.jsonl",
            out_dir / "prepare_data_summary.json",
        ]
        for path in expected:
            assert path.exists(), f"missing artifact: {path}"
        if importlib.util.find_spec("pyarrow") is not None:
            assert (out_dir / "datapool.normalized.parquet").exists()
            assert (out_dir / "datapool.train.parquet").exists()
            assert (out_dir / "datapool.eval.parquet").exists()
            assert (out_dir / "datapool.train.sampled.parquet").exists()

        normalized_rows = read_jsonl(out_dir / "datapool.normalized.jsonl")
        train_rows = read_jsonl(out_dir / "datapool.train.jsonl")
        eval_rows = read_jsonl(out_dir / "datapool.eval.jsonl")
        sampled_rows = read_jsonl(out_dir / "datapool.train.sampled.jsonl")
        validate_artifact_rows(normalized_rows, "normalized")
        validate_artifact_rows(train_rows, "normalized")
        validate_artifact_rows(eval_rows, "normalized")
        validate_artifact_rows(sampled_rows, "normalized")

        train_ids = {row["id"] for row in train_rows}
        eval_ids = {row["id"] for row in eval_rows}
        sampled_ids = [row["id"] for row in sampled_rows]

        assert train_ids.isdisjoint(eval_ids)
        assert set(sampled_ids).issubset(train_ids)
        assert len(sampled_rows) == summary["sampled_rows"]

        summary_file = json.loads((out_dir / "prepare_data_summary.json").read_text(encoding="utf-8"))
        assert summary_file["artifact_dir"].endswith("artifacts/data")
    finally:
        os.chdir(old_cwd)


def test_prepare_data_overflow_split_creates_chunk_ids(tmp_path: Path) -> None:
    workdir = tmp_path / "work_split"
    workdir.mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd()
    try:
        os.chdir(workdir)
        run_prepare_data(
            config_path=str(ROOT / "configs" / "scp_stage4.yaml"),
            overrides=[
                "data.length.max_source_tokens=3",
                "data.length.overflow=split",
                "data.length.split.max_chunks_per_row=2",
                "data.length.split.fallback_for_long_sentence=split",
                "data.subset_size=8",
            ],
        )
        rows = read_jsonl(workdir / "artifacts" / "data" / "datapool.normalized.jsonl")
        assert any("__chunk_" in str(row["id"]) for row in rows)
    finally:
        os.chdir(old_cwd)


def test_split_long_source_uses_batched_sentence_counts() -> None:
    row = {
        "id": "row-001",
        "source": "One short sentence. Another short sentence. Third sentence. Fourth sentence.",
        "metadata": {"parent_id": None, "chunk_idx": None},
    }
    exact_calls = {"count": 0}

    def _token_count(text: str) -> int:
        exact_calls["count"] += 1
        return len(text.split())

    def _token_count_batch(texts: list[str]) -> list[int]:
        return [len(text.split()) for text in texts]

    chunks = prepare_data_module._split_long_source(
        row,
        token_count=_token_count,
        token_count_batch=_token_count_batch,
        max_tokens_per_chunk=8,
        max_chunks=8,
        fallback_for_long_sentence="split",
        on_max_chunks_exceeded="skip",
    )
    assert chunks
    # Sentence token costs are batched, so exact per-string counts should stay near chunk count.
    assert exact_calls["count"] <= len(chunks) + 1


def test_split_long_source_keeps_first_chunks_when_cap_exceeded() -> None:
    row = {
        "id": "row-cap",
        "source": "One. Two. Three. Four. Five.",
        "metadata": {"parent_id": None, "chunk_idx": None},
    }

    chunks = prepare_data_module._split_long_source(
        row,
        token_count=lambda text: len(text.split()),
        token_count_batch=lambda texts: [len(text.split()) for text in texts],
        max_tokens_per_chunk=1,
        max_chunks=3,
        fallback_for_long_sentence="split",
        on_max_chunks_exceeded="keep_first",
    )

    assert [chunk["id"] for chunk in chunks] == [
        "row-cap__chunk_0",
        "row-cap__chunk_1",
        "row-cap__chunk_2",
    ]
    assert [chunk["source"] for chunk in chunks] == ["One.", "Two.", "Three."]


def test_prepare_data_local_jsonl_runtime_uses_configured_source(tmp_path: Path) -> None:
    workdir = tmp_path / "work_local_jsonl"
    workdir.mkdir(parents=True, exist_ok=True)
    raw_path = workdir / "raw.jsonl"
    raw_path.write_text(
        json.dumps(
            {
                "id": "local-row-1",
                "dataset": "local_jsonl_dataset",
                "source_text": "A configured local JSONL row.",
                "title": "Configured",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    old_cwd = Path.cwd()
    try:
        os.chdir(workdir)
        run_prepare_data(
            config_path=str(ROOT / "configs" / "scp_stage4.yaml"),
            overrides=[
                "data.runtime.mode=local_jsonl",
                f"data.runtime.local_jsonl_path={raw_path}",
                "data.subset_size=1",
                "data.split.eval_ratio=0",
            ],
        )
        rows = read_jsonl(workdir / "artifacts" / "data" / "datapool.normalized.jsonl")
        assert rows
        assert rows[0]["id"].startswith("local-row-1")
        assert rows[0]["dataset"] == "local_jsonl_dataset"
    finally:
        os.chdir(old_cwd)


def test_prepare_data_can_force_jsonl_intermediate_format(tmp_path: Path) -> None:
    workdir = tmp_path / "work_jsonl_intermediate"
    workdir.mkdir(parents=True, exist_ok=True)

    old_cwd = Path.cwd()
    try:
        os.chdir(workdir)
        run_prepare_data(
            config_path=str(ROOT / "configs" / "scp_stage4.yaml"),
            overrides=[
                "data.runtime.prepare_data.intermediate_format=jsonl",
                "data.split.eval_ratio=0",
                "data.subset_size=8",
            ],
        )
        out_dir = workdir / "artifacts" / "data"
        assert (out_dir / "datapool.normalized.jsonl").exists()
        assert not (out_dir / ".prepare_data.normalized.tmp.jsonl").exists()
    finally:
        os.chdir(old_cwd)


def test_prepare_data_emits_progress_logs(tmp_path: Path, capsys) -> None:
    workdir = tmp_path / "work_progress"
    workdir.mkdir(parents=True, exist_ok=True)

    old_cwd = Path.cwd()
    try:
        os.chdir(workdir)
        run_prepare_data(
            config_path=str(ROOT / "configs" / "scp_stage4.yaml"),
            overrides=[
                "data.split.eval_ratio=0",
                "data.subset_size=2",
                "data.runtime.prepare_data.progress_enabled=true",
                "data.runtime.prepare_data.progress_every_rows=1",
                "data.runtime.prepare_data.progress_every_seconds=0.001",
            ],
        )
        captured = capsys.readouterr()
        assert "phase=normalize" in captured.err
        assert "phase=normalize-summary" in captured.err
        assert "phase=split" in captured.err
        assert "rows_per_sec=" in captured.err
    finally:
        os.chdir(old_cwd)


def test_prepare_data_streaming_split_keeps_exact_eval_count(tmp_path: Path) -> None:
    workdir = tmp_path / "work_stream_split"
    workdir.mkdir(parents=True, exist_ok=True)
    raw_path = workdir / "raw.jsonl"
    rows = []
    for idx in range(10):
        rows.append(
            json.dumps(
                {
                    "id": f"row-{idx:04d}",
                    "dataset": "stream_split_dataset",
                    "source_text": f"Streaming row {idx}",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    raw_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    old_cwd = Path.cwd()
    try:
        os.chdir(workdir)
        run_prepare_data(
            config_path=str(ROOT / "configs" / "scp_stage4.yaml"),
            overrides=[
                "data.runtime.mode=local_jsonl",
                f"data.runtime.local_jsonl_path={raw_path}",
                "data.split.eval_ratio=0.2",
                "data.sampling.strategy=first_n",
                "data.subset_size=3",
            ],
        )
        out_dir = workdir / "artifacts" / "data"
        train_rows = read_jsonl(out_dir / "datapool.train.jsonl")
        eval_rows = read_jsonl(out_dir / "datapool.eval.jsonl")
        sampled_rows = read_jsonl(out_dir / "datapool.train.sampled.jsonl")
        assert len(eval_rows) == 2
        assert len(train_rows) == 8
        assert len(sampled_rows) == 3
    finally:
        os.chdir(old_cwd)


def test_materialize_duplicate_file_falls_back_to_copy(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    target = tmp_path / "sampled.jsonl"
    source.write_text('{"id":"x"}\n', encoding="utf-8")

    def _raise_exdev(src: Path, dst: Path) -> None:
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(prepare_data_module.os, "link", _raise_exdev)
    mode = prepare_data_module._materialize_duplicate_file(source, target)
    assert mode == "copy"
    assert target.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_prepare_data_summary_includes_length_policy_skip_counts(tmp_path: Path) -> None:
    workdir = tmp_path / "work_length_policy_summary"
    workdir.mkdir(parents=True, exist_ok=True)
    raw_path = workdir / "raw.jsonl"

    long_sentence = " ".join(f"tok{i}" for i in range(220))
    raw_path.write_text(
        json.dumps(
            {"id": "row-skip", "source_text": long_sentence},
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
        + json.dumps(
            {"id": "row-keep", "source_text": "short text kept"},
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    old_cwd = Path.cwd()
    try:
        os.chdir(workdir)
        summary = run_prepare_data(
            config_path=str(ROOT / "configs" / "scp_stage4.yaml"),
            overrides=[
                "data.runtime.mode=local_jsonl",
                f"data.runtime.local_jsonl_path={raw_path}",
                "data.split.eval_ratio=0",
                "data.subset_size=2",
                "data.length.max_source_tokens=32",
                "data.length.max_total_tokens=128",
                "data.length.prompt_template_tokens=8",
                "data.length.min_available_output_tokens=8",
                "data.length.safety_margin_tokens=8",
                "data.length.overflow=split",
                "data.length.split.max_source_tokens_per_chunk=32",
                "data.length.split.max_chunks_per_row=4",
                "data.length.split.fallback_for_long_sentence=skip",
                "data.length.split.on_max_chunks_exceeded=skip",
                "data.length.mode=whitespace",
            ],
        )
        length_policy = summary["length_policy"]
        assert length_policy["input_rows"] == 2
        assert length_policy["output_rows"] == 1
        assert length_policy["skipped_long_sentence"] == 1
        assert length_policy["skipped_total"] == 1
    finally:
        os.chdir(old_cwd)


def test_prepare_data_length_policy_uses_source_limit_only(tmp_path: Path) -> None:
    workdir = tmp_path / "work_source_only_length"
    workdir.mkdir(parents=True, exist_ok=True)
    raw_path = workdir / "raw.jsonl"
    raw_path.write_text(
        json.dumps(
            {"id": "row-keep", "source_text": "one two three four five"},
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    old_cwd = Path.cwd()
    try:
        os.chdir(workdir)
        summary = run_prepare_data(
            config_path=str(ROOT / "configs" / "scp_stage4.yaml"),
            overrides=[
                "data.runtime.mode=local_jsonl",
                f"data.runtime.local_jsonl_path={raw_path}",
                "data.split.eval_ratio=0",
                "data.subset_size=1",
                "data.length.max_source_tokens=8",
                "data.length.max_total_tokens=10",
                "data.length.prompt_template_tokens=4",
                "data.length.min_available_output_tokens=4",
                "data.length.safety_margin_tokens=0",
                "data.length.overflow=split",
                "data.length.split.max_source_tokens_per_chunk=8",
                "data.length.split.min_chunk_tokens=1",
                "data.length.mode=whitespace",
            ],
        )
        length_policy = summary["length_policy"]
        assert length_policy["input_rows"] == 1
        assert length_policy["output_rows"] == 1
        assert length_policy["split_input_rows"] == 0
        assert length_policy["skipped_total"] == 0
    finally:
        os.chdir(old_cwd)


def test_prepare_data_hf_runtime_falls_back_to_snapshot_jsonl(tmp_path: Path, monkeypatch) -> None:
    workdir = tmp_path / "work_hf_fallback"
    workdir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = workdir / "snapshot"
    data_dir = snapshot_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "reuter_processed.jsonl").write_text(
        json.dumps(
            {
                "source_text": "First fallback row.",
                "title": "A title",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
        + json.dumps(
            {
                "source_text": "Second fallback row.",
                "title": "Another title",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    def _raise_cast_error(*args, **kwargs):
        raise RuntimeError("Couldn't cast array of type struct to schema")

    fake_datasets = types.SimpleNamespace(load_dataset=_raise_cast_error)
    fake_hub = types.SimpleNamespace(
        snapshot_download=lambda **kwargs: str(snapshot_dir),
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    old_cwd = Path.cwd()
    try:
        os.chdir(workdir)
        run_prepare_data(
            config_path=str(ROOT / "configs" / "scp_stage4_real.yaml"),
            overrides=[
                "data.runtime.mode=hf",
                "data.datasets=[{\"name\":\"alwaysgood/reuter_processed\",\"split\":\"train\"}]",
                "data.split.eval_ratio=0",
                "data.subset_size=2",
                "data.length.tokenizer_fallback=whitespace",
            ],
        )
        rows = read_jsonl(workdir / "artifacts" / "data" / "datapool.normalized.jsonl")
        assert len(rows) == 2
        assert rows[0]["dataset"] == "alwaysgood/reuter_processed"
        assert rows[0]["id"].startswith("alwaysgood/reuter_processed:")
    finally:
        os.chdir(old_cwd)


def test_prepare_data_hf_runtime_uses_num_workers_as_num_proc(tmp_path: Path, monkeypatch) -> None:
    workdir = tmp_path / "work_hf_num_proc"
    workdir.mkdir(parents=True, exist_ok=True)
    captured: dict[str, object] = {}

    def _fake_load_dataset(*args, **kwargs):
        captured["num_proc"] = kwargs.get("num_proc")
        return [
            {
                "id": "row-0001",
                "dataset": "alwaysgood/reuter_processed",
                "source_text": "A single row for num_proc test.",
            }
        ]

    fake_datasets = types.SimpleNamespace(load_dataset=_fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    old_cwd = Path.cwd()
    try:
        os.chdir(workdir)
        run_prepare_data(
            config_path=str(ROOT / "configs" / "scp_stage4_real.yaml"),
            overrides=[
                "data.runtime.mode=hf",
                "data.datasets=[{\"name\":\"alwaysgood/reuter_processed\",\"split\":\"train\"}]",
                "data.split.eval_ratio=0",
                "data.subset_size=1",
                "data.num_workers=10",
            ],
        )
        assert captured.get("num_proc") == 10
    finally:
        os.chdir(old_cwd)


def test_prepare_data_hf_runtime_parallel_dataset_download_preserves_dataset_order(
    tmp_path: Path, monkeypatch
) -> None:
    workdir = tmp_path / "work_hf_parallel_datasets"
    workdir.mkdir(parents=True, exist_ok=True)

    def _fake_load_dataset(name, *args, **kwargs):
        if "dataset_a" in name:
            time.sleep(0.05)
        return [
            {
                "source_text": f"row for {name}",
            }
        ]

    fake_datasets = types.SimpleNamespace(load_dataset=_fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    old_cwd = Path.cwd()
    try:
        os.chdir(workdir)
        run_prepare_data(
            config_path=str(ROOT / "configs" / "scp_stage4_real.yaml"),
            overrides=[
                "data.runtime.mode=hf",
                "data.datasets=[{\"name\":\"local/dataset_a\",\"split\":\"train\"},{\"name\":\"local/dataset_b\",\"split\":\"train\"}]",
                "data.runtime.hf.dataset_download_workers=2",
                "data.num_workers=1",
                "data.split.eval_ratio=0",
                "data.subset_size=2",
            ],
        )
        rows = read_jsonl(workdir / "artifacts" / "data" / "datapool.normalized.jsonl")
        assert len(rows) == 2
        assert rows[0]["dataset"] == "local/dataset_a"
        assert rows[1]["dataset"] == "local/dataset_b"
    finally:
        os.chdir(old_cwd)
