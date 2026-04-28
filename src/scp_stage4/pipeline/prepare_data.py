"""Local prepare-data implementation for contract harness."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from scp_stage4.config.loader import compose_config
from scp_stage4.config.validator import validate_config
from scp_stage4.data import read_jsonl, write_jsonl
from scp_stage4.schema import validate_artifact_rows


class PrepareDataError(RuntimeError):
    """Raised when prepare-data contract cannot be satisfied."""


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _get_by_dotpath(data: Mapping[str, Any], key: str) -> Any:
    cursor: Any = data
    for part in key.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _first_non_empty(values: Iterable[Any]) -> str | None:
    for value in values:
        if isinstance(value, str):
            normalized = _normalize_whitespace(value)
            if normalized:
                return normalized
    return None


def _infer_document_type(dataset_name: str, row: Mapping[str, Any]) -> str | None:
    existing = _get_by_dotpath(row, "metadata.document_type")
    if isinstance(existing, str) and existing in {"article", "filing", "earnings_call", "other"}:
        return existing

    lowered = dataset_name.lower()
    if "reuter" in lowered or "bloomberg" in lowered:
        return "article"
    if "10k" in lowered or "sec" in lowered or "filing" in lowered:
        return "filing"
    if "earnings" in lowered or "call" in lowered:
        return "earnings_call"
    return "other"


def _fallback_raw_rows(dataset_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(128):
        rows.append(
            {
                "id": f"{dataset_name}:{idx:06d}",
                "dataset": dataset_name,
                "source_text": f"Fixture sentence number {idx} for checks.",
                "title": f"{dataset_name} title {idx}",
            }
        )
    return rows


def _fixture_raw_rows(data_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        Path("tests/fixtures/raw.train.jsonl"),
        Path("tests/fixtures/input.happy.jsonl"),
    ]
    for path in candidates:
        if path.exists():
            rows = read_jsonl(path)
            return [dict(row) for row in rows]

    dataset_name = "local_fixture_dataset"
    datasets = data_cfg.get("datasets")
    if isinstance(datasets, list) and datasets:
        first = datasets[0]
        if isinstance(first, Mapping):
            dataset_name = str(first.get("name", dataset_name))
    return _fallback_raw_rows(dataset_name)


def _load_local_jsonl_rows(data_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    runtime_cfg = data_cfg.get("runtime", {})
    if not isinstance(runtime_cfg, Mapping):
        raise PrepareDataError("data.runtime must be a mapping")
    path_value = runtime_cfg.get("local_jsonl_path")
    if not isinstance(path_value, str) or not path_value.strip():
        raise PrepareDataError(
            "data.runtime.local_jsonl_path is required when data.runtime.mode=local_jsonl"
        )
    path = Path(path_value)
    if not path.exists():
        raise PrepareDataError(f"local JSONL dataset not found: {path}")
    return [dict(row) for row in read_jsonl(path)]


def _load_hf_rows(data_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ModuleNotFoundError as exc:
        raise PrepareDataError(
            "data.runtime.mode=hf requires the Hugging Face 'datasets' package"
        ) from exc

    runtime_cfg = data_cfg.get("runtime", {})
    if not isinstance(runtime_cfg, Mapping):
        runtime_cfg = {}
    hf_cfg = runtime_cfg.get("hf", {})
    if not isinstance(hf_cfg, Mapping):
        hf_cfg = {}

    streaming = bool(hf_cfg.get("streaming", False))
    max_rows_raw = hf_cfg.get("max_rows_per_dataset")
    max_rows_per_dataset = int(max_rows_raw) if max_rows_raw is not None else None

    dataset_specs = data_cfg.get("datasets")
    if not isinstance(dataset_specs, list) or not dataset_specs:
        raise PrepareDataError("data.datasets must be a non-empty list for HF loading")

    loaded_rows: list[dict[str, Any]] = []
    for dataset_index, spec in enumerate(dataset_specs):
        if not isinstance(spec, Mapping):
            raise PrepareDataError(f"data.datasets[{dataset_index}] must be a mapping")
        name = str(spec.get("name", "")).strip()
        if not name:
            raise PrepareDataError(f"data.datasets[{dataset_index}].name is required")
        split = str(spec.get("split", "train"))

        load_kwargs: dict[str, Any] = {
            "split": split,
            "streaming": streaming,
        }
        for optional_key in (
            "data_dir",
            "data_files",
            "revision",
            "trust_remote_code",
        ):
            if optional_key in spec:
                load_kwargs[optional_key] = spec[optional_key]

        config_name = spec.get("config_name")
        if isinstance(config_name, str) and config_name.strip():
            dataset = load_dataset(name, config_name, **load_kwargs)
        else:
            dataset = load_dataset(name, **load_kwargs)

        for row_index, raw_row in enumerate(dataset):
            if max_rows_per_dataset is not None and row_index >= max_rows_per_dataset:
                break
            if not isinstance(raw_row, Mapping):
                continue
            row = dict(raw_row)
            row.setdefault("dataset", name)
            raw_id = row.get("id") or row.get("_id") or row.get("doc_id")
            row["id"] = str(raw_id) if raw_id is not None else f"{name}:{row_index:08d}"
            loaded_rows.append(row)

    if not loaded_rows:
        raise PrepareDataError("HF loading produced zero rows")
    return loaded_rows


def _load_raw_rows(data_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    runtime_cfg = data_cfg.get("runtime", {})
    if not isinstance(runtime_cfg, Mapping):
        runtime_cfg = {}
    mode = str(runtime_cfg.get("mode", "fixture"))
    if mode == "fixture":
        return _fixture_raw_rows(data_cfg)
    if mode == "local_jsonl":
        return _load_local_jsonl_rows(data_cfg)
    if mode == "hf":
        return _load_hf_rows(data_cfg)
    raise PrepareDataError(f"Unsupported data.runtime.mode: {mode}")


def _normalize_rows(raw_rows: list[dict[str, Any]], data_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    dataset_default = "local_fixture_dataset"
    datasets = data_cfg.get("datasets")
    if isinstance(datasets, list) and datasets:
        first = datasets[0]
        if isinstance(first, Mapping):
            dataset_default = str(first.get("name", dataset_default))

    source_columns = data_cfg.get("text_columns", ["source_text", "text", "source"])
    if not isinstance(source_columns, list) or not source_columns:
        source_columns = ["source_text", "text", "source"]
    else:
        source_columns = [str(col) for col in source_columns]
        if "source" not in source_columns:
            # Accept already-normalized fixture rows in local harness.
            source_columns.append("source")

    title_candidates = [
        "metadata.Headline",
        "metadata.headline",
        "metadata.title",
        "Headline",
        "headline",
        "title",
    ]

    translatable_fields = data_cfg.get("translatable_fields")
    normalized: list[dict[str, Any]] = []

    for index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            continue
        dataset_name = str(row.get("dataset", dataset_default))
        base_id = str(row.get("id", f"{dataset_name}:{index:06d}"))
        source = _first_non_empty(_get_by_dotpath(row, key) for key in source_columns)
        title = _first_non_empty(_get_by_dotpath(row, key) for key in title_candidates)
        document_type = _infer_document_type(dataset_name, row)

        emitted = False
        if isinstance(translatable_fields, list) and translatable_fields:
            for field in translatable_fields:
                if not isinstance(field, Mapping):
                    continue
                field_name = str(field.get("name", "field"))
                columns = field.get("columns", [])
                if not isinstance(columns, list) or not columns:
                    continue
                value = _first_non_empty(_get_by_dotpath(row, col) for col in columns)
                if value is None:
                    optional = bool(field.get("optional", False))
                    if optional:
                        continue
                    continue
                text_role = str(field.get("text_role", "other"))
                normalized.append(
                    {
                        "id": f"{base_id}__{field_name}",
                        "dataset": dataset_name,
                        "source": value,
                        "metadata": {
                            "title": title,
                            "document_type": document_type,
                            "text_role": text_role,
                            "original_id": base_id,
                            "parent_id": None,
                            "chunk_idx": None,
                        },
                    }
                )
                emitted = True

        if emitted:
            continue
        if source is None:
            continue

        normalized.append(
            {
                "id": base_id,
                "dataset": dataset_name,
                "source": source,
                "metadata": {
                    "title": title,
                    "document_type": document_type,
                    "text_role": "body",
                    "original_id": base_id,
                    "parent_id": None,
                    "chunk_idx": None,
                },
            }
        )

    return validate_artifact_rows(normalized, "normalized")


def _estimate_tokens(text: str) -> int:
    return len(text.split())


def _build_token_counter(cfg: Mapping[str, Any]) -> Callable[[str], int]:
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, Mapping):
        data_cfg = {}
    length_cfg = data_cfg.get("length", {})
    if not isinstance(length_cfg, Mapping):
        length_cfg = {}

    mode = str(length_cfg.get("mode", "whitespace"))
    if mode != "tokenizer":
        return _estimate_tokens

    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, Mapping):
        model_cfg = {}
    tokenizer_name = length_cfg.get("tokenizer_name") or model_cfg.get("name")
    fallback = str(length_cfg.get("tokenizer_fallback", "whitespace"))
    local_files_only = bool(length_cfg.get("tokenizer_local_files_only", False))
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))

    try:
        from transformers import AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_name),
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        if fallback == "whitespace":
            return _estimate_tokens
        raise PrepareDataError(
            "tokenizer length mode requires a loadable Hugging Face tokenizer; "
            f"failed to load {tokenizer_name!r}"
        ) from exc

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count_tokens


_SENTENCE_RE = re.compile(r"[^.!?。！？\n]+[.!?。！？]?(?:\s+|$)|[^\n]+(?:\n|$)")


def _sentence_units(text: str) -> list[str]:
    units = [_normalize_whitespace(match.group(0)) for match in _SENTENCE_RE.finditer(text)]
    return [unit for unit in units if unit]


def _split_long_source(
    row: Mapping[str, Any],
    *,
    token_count: Callable[[str], int],
    max_tokens_per_chunk: int,
    max_chunks: int,
    fallback_for_long_sentence: str,
    on_max_chunks_exceeded: str,
) -> list[dict[str, Any]]:
    source = str(row["source"])
    chunks: list[str] = []
    current: list[str] = []

    def flush_current() -> None:
        if current:
            chunks.append(_normalize_whitespace(" ".join(current)))
            current.clear()

    for sentence in _sentence_units(source):
        sentence_tokens = token_count(sentence)
        if sentence_tokens > max_tokens_per_chunk:
            flush_current()
            if fallback_for_long_sentence == "truncate":
                chunks.append(" ".join(sentence.split()[:max_tokens_per_chunk]))
            elif fallback_for_long_sentence == "split":
                words = sentence.split()
                for start in range(0, len(words), max_tokens_per_chunk):
                    chunks.append(" ".join(words[start : start + max_tokens_per_chunk]))
            else:
                return []
            continue

        candidate = _normalize_whitespace(" ".join([*current, sentence]))
        if current and token_count(candidate) > max_tokens_per_chunk:
            flush_current()
        current.append(sentence)
    flush_current()

    chunks = [chunk for chunk in chunks if chunk]
    if len(chunks) > max_chunks:
        if on_max_chunks_exceeded == "error":
            raise PrepareDataError(f"row {row['id']} exceeded max_chunks_per_row={max_chunks}")
        return []

    out_rows: list[dict[str, Any]] = []
    for chunk_idx, chunk in enumerate(chunks):
        out = dict(row)
        out["id"] = f"{row['id']}__chunk_{chunk_idx}"
        out["source"] = chunk
        metadata = dict(row.get("metadata", {}))
        metadata["parent_id"] = row["id"]
        metadata["chunk_idx"] = chunk_idx
        out["metadata"] = metadata
        out_rows.append(out)
    return out_rows


def _resolved_runtime_token_limits(cfg: Mapping[str, Any]) -> tuple[int, int]:
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, Mapping):
        data_cfg = {}
    length_cfg = data_cfg.get("length", {})
    if not isinstance(length_cfg, Mapping):
        length_cfg = {}

    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, Mapping):
        model_cfg = {}

    model_max_length = int(model_cfg.get("max_length", 0))
    model_max_seq_length = int(model_cfg.get("max_seq_length", model_max_length))
    max_total_tokens = int(length_cfg.get("max_total_tokens", model_max_length))
    runtime_max_total = min(model_max_length, model_max_seq_length, max_total_tokens)

    prompt_template_tokens = int(length_cfg.get("prompt_template_tokens", 0))
    safety_margin_tokens = int(length_cfg.get("safety_margin_tokens", 0))
    min_available_output_tokens = int(length_cfg.get("min_available_output_tokens", 0))
    max_source_tokens = int(length_cfg.get("max_source_tokens", 0))

    source_budget_by_context = (
        runtime_max_total
        - prompt_template_tokens
        - safety_margin_tokens
        - min_available_output_tokens
    )
    effective_max_source_tokens = min(max_source_tokens, source_budget_by_context)
    if effective_max_source_tokens <= 0:
        raise PrepareDataError(
            "length policy is unsatisfiable: "
            "max_total_tokens - prompt_template_tokens - safety_margin_tokens - "
            "min_available_output_tokens must be > 0"
        )
    return runtime_max_total, effective_max_source_tokens


def _apply_length_policy(rows: list[dict[str, Any]], cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, Mapping):
        data_cfg = {}
    length_cfg = data_cfg.get("length", {})
    if not isinstance(length_cfg, Mapping) or not bool(length_cfg.get("enabled", True)):
        return rows

    runtime_max_total, effective_max_source_tokens = _resolved_runtime_token_limits(cfg)
    prompt_template_tokens = int(length_cfg.get("prompt_template_tokens", 0))
    safety_margin_tokens = int(length_cfg.get("safety_margin_tokens", 0))
    min_available_output_tokens = int(length_cfg.get("min_available_output_tokens", 0))
    overflow = str(length_cfg.get("overflow", "split"))
    split_cfg = length_cfg.get("split", {})
    if not isinstance(split_cfg, Mapping):
        split_cfg = {}
    max_chunks = int(split_cfg.get("max_chunks_per_row", 4))
    max_tokens_per_chunk = int(
        split_cfg.get("max_source_tokens_per_chunk", effective_max_source_tokens)
    )
    max_tokens_per_chunk = min(max_tokens_per_chunk, effective_max_source_tokens)
    fallback_for_long_sentence = str(split_cfg.get("fallback_for_long_sentence", "skip"))
    on_max_chunks_exceeded = str(split_cfg.get("on_max_chunks_exceeded", "skip"))
    token_count = _build_token_counter(cfg)

    filtered: list[dict[str, Any]] = []
    for row in rows:
        source = str(row["source"])
        source_tokens = token_count(source)
        available_output_budget = (
            runtime_max_total
            - prompt_template_tokens
            - source_tokens
            - safety_margin_tokens
        )
        if (
            source_tokens <= effective_max_source_tokens
            and available_output_budget >= min_available_output_tokens
        ):
            filtered.append(row)
            continue

        if overflow == "skip":
            continue

        if overflow == "truncate":
            words = source.split()
            truncated = " ".join(words[:effective_max_source_tokens])
            out = dict(row)
            out["source"] = truncated
            truncated_tokens = token_count(truncated)
            available_after_truncation = (
                runtime_max_total
                - prompt_template_tokens
                - truncated_tokens
                - safety_margin_tokens
            )
            if available_after_truncation >= min_available_output_tokens:
                filtered.append(out)
            continue

        filtered.extend(
            _split_long_source(
                row,
                token_count=token_count,
                max_tokens_per_chunk=max_tokens_per_chunk,
                max_chunks=max_chunks,
                fallback_for_long_sentence=fallback_for_long_sentence,
                on_max_chunks_exceeded=on_max_chunks_exceeded,
            )
        )

    return validate_artifact_rows(filtered, "normalized")


def _split_train_eval(
    rows: list[dict[str, Any]], eval_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    total = len(rows)
    if total == 0:
        return [], []
    if total == 1 or eval_ratio <= 0:
        return list(rows), []

    eval_count = int(math.ceil(total * eval_ratio))
    eval_count = max(1, min(eval_count, total - 1))

    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    eval_set = set(indices[:eval_count])

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if idx in eval_set:
            eval_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, eval_rows


def _sample_train_rows(
    train_rows: list[dict[str, Any]],
    subset_size: int | None,
    strategy: str,
    seed: int,
) -> list[dict[str, Any]]:
    if subset_size is None:
        return list(train_rows)
    if not train_rows:
        return []

    size = max(1, min(int(subset_size), len(train_rows)))
    if strategy == "random":
        indices = list(range(len(train_rows)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        selected = sorted(indices[:size])
        return [train_rows[i] for i in selected]
    # first_n default
    return list(train_rows[:size])


def _write_ood_placeholder(data_cfg: Mapping[str, Any], out_path: Path) -> None:
    ood_cfg = data_cfg.get("ood_test", {})
    if not isinstance(ood_cfg, Mapping) or not bool(ood_cfg.get("enabled", False)):
        return

    source_path = Path(str(ood_cfg.get("path", "")))
    if not source_path.exists():
        write_jsonl(out_path, [])
        return

    source_col = str(ood_cfg.get("source_column", "Source_En"))
    target_col = str(ood_cfg.get("target_column", "Target_Ko"))
    rows: list[dict[str, Any]] = []
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            source = _normalize_whitespace(str(row.get(source_col, "")))
            target = _normalize_whitespace(str(row.get(target_col, "")))
            if not source:
                continue
            rows.append(
                {
                    "id": f"ood_{idx:06d}",
                    "source": source,
                    "target": target,
                }
            )
    write_jsonl(out_path, rows)


def run_prepare_data(
    config_path: str = "configs/scp_stage4.yaml",
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    cfg = compose_config(config_path, overrides=overrides)
    validate_config(cfg)

    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, Mapping):
        raise PrepareDataError("data config must be a mapping")

    raw_rows = _load_raw_rows(data_cfg)
    normalized = _normalize_rows(raw_rows, data_cfg)
    normalized = _apply_length_policy(normalized, cfg)

    split_cfg = data_cfg.get("split", {})
    if not isinstance(split_cfg, Mapping):
        split_cfg = {}
    eval_ratio = float(split_cfg.get("eval_ratio", 0.02))
    split_seed = int(split_cfg.get("seed", 42))
    train_rows, eval_rows = _split_train_eval(normalized, eval_ratio, split_seed)

    sampling_cfg = data_cfg.get("sampling", {})
    if not isinstance(sampling_cfg, Mapping):
        sampling_cfg = {}
    sampling_strategy = str(sampling_cfg.get("strategy", "first_n"))
    sampling_seed = int(sampling_cfg.get("seed", 42))
    subset_size = data_cfg.get("subset_size")
    sampled_rows = _sample_train_rows(
        train_rows,
        int(subset_size) if subset_size is not None else None,
        sampling_strategy,
        sampling_seed,
    )

    out_dir = Path("artifacts/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "datapool.normalized.jsonl", normalized)
    write_jsonl(out_dir / "datapool.train.jsonl", train_rows)
    write_jsonl(out_dir / "datapool.eval.jsonl", eval_rows)
    write_jsonl(out_dir / "datapool.train.sampled.jsonl", sampled_rows)
    _write_ood_placeholder(data_cfg, out_dir / "ood_test.jsonl")

    summary = {
        "normalized_rows": len(normalized),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "sampled_rows": len(sampled_rows),
        "artifact_dir": str(out_dir),
    }
    (out_dir / "prepare_data_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare local data artifacts")
    parser.add_argument("--config", default="configs/scp_stage4.yaml")
    args, overrides = parser.parse_known_args(argv)
    summary = run_prepare_data(config_path=args.config, overrides=overrides)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
