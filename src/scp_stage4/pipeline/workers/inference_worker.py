"""Real inference worker for subprocess runtime.

This worker executes local model generation for infer-q1 / infer-q2 using
Transformers and optional PEFT adapters.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch

from scp_stage4.data import read_jsonl, write_jsonl
from scp_stage4.pipeline.prompting import (
    PromptConfigError,
    render_translation_prompt,
    sft_response_template,
)
from scp_stage4.pipeline.workers.common import (
    WorkerContractError,
    parse_worker_args,
    validate_phase_request_rows,
    validate_phase_response_rows,
)

_NOOP_COLLAPSE_MARKER = "NOOP_COLLAPSE.json"


@dataclass(frozen=True)
class _ModelRuntime:
    model_ref: str
    tokenizer_ref: str
    trust_remote_code: bool
    torch_dtype: torch.dtype | str | None
    max_seq_length: int | None
    padding_side: str


@dataclass(frozen=True)
class _ThroughputRuntime:
    strategy: str
    max_batch_tokens: int
    pad_to_multiple_of: int | None
    preserve_order: bool
    restore_order_in_artifacts: bool


@dataclass(frozen=True)
class _UnslothRuntime:
    enabled: bool
    fallback_to_transformers: bool


@dataclass(frozen=True)
class _BatchItem:
    order_idx: int
    request: Mapping[str, Any]
    prompt: str
    prompt_tokens: int


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dtype_from_config(dtype_value: Any) -> torch.dtype | str | None:
    if dtype_value is None:
        return None
    text = str(dtype_value).strip().lower()
    if text in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if text in {"fp16", "float16", "half"}:
        return torch.float16
    if text in {"fp32", "float32"}:
        return torch.float32
    if text in {"auto", ""}:
        return "auto"
    return None


def _is_lora_adapter_path(path: Path) -> bool:
    return path.exists() and (path / "adapter_config.json").exists()


def _is_noop_collapse_adapter_path(path: Path) -> bool:
    return path.exists() and (path / _NOOP_COLLAPSE_MARKER).exists()


def _is_model_checkpoint_path(path: Path) -> bool:
    if not path.exists():
        return False
    return (path / "config.json").exists() or (path / "model.safetensors").exists()


def _resolve_runtime(request: Mapping[str, Any]) -> _ModelRuntime:
    runtime_cfg = _as_dict(request.get("runtime_config"))
    model_cfg = _as_dict(runtime_cfg.get("model"))
    model_name = str(model_cfg.get("name", "")).strip()
    if not model_name:
        raise WorkerContractError("inference request runtime_config.model.name is required")

    base_checkpoint = request.get("base_checkpoint")
    model_ref = model_name
    if isinstance(base_checkpoint, str) and base_checkpoint.strip():
        cp = Path(base_checkpoint)
        if _is_model_checkpoint_path(cp):
            model_ref = str(cp)
        elif cp.exists() and not _is_lora_adapter_path(cp):
            model_ref = str(cp)

    tokenizer_ref = model_name
    if isinstance(model_ref, str) and Path(model_ref).exists() and _is_model_checkpoint_path(Path(model_ref)):
        tokenizer_ref = model_ref

    max_seq_length = model_cfg.get("max_seq_length") or model_cfg.get("max_length")
    if isinstance(max_seq_length, bool) or not isinstance(max_seq_length, int):
        max_seq_length = 8192

    padding_side = str(model_cfg.get("padding_side", "right"))
    if padding_side not in {"left", "right"}:
        padding_side = "right"

    return _ModelRuntime(
        model_ref=model_ref,
        tokenizer_ref=tokenizer_ref,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        torch_dtype=_dtype_from_config(model_cfg.get("dtype")),
        max_seq_length=max_seq_length,
        padding_side=padding_side,
    )


def _build_prompt(source: str, request: Mapping[str, Any] | None = None) -> str:
    if request is None:
        return (
            "You are a professional English to Korean translator.\n"
            "Translate the English input into natural Korean.\n"
            "Return only the Korean translation.\n\n"
            f"English:\n{source}\n\nKorean:\n"
        )

    runtime_cfg = _as_dict(request.get("runtime_config"))
    prompts_cfg = _as_dict(runtime_cfg.get("prompts"))
    row_id = str(request.get("row_id") or request.get("id") or "")
    subset_idx_raw = request.get("subset_idx")
    subset_idx: int | None = None
    if isinstance(subset_idx_raw, int) and not isinstance(subset_idx_raw, bool):
        subset_idx = subset_idx_raw

    try:
        prompt, _ = render_translation_prompt(
            prompts=prompts_cfg,
            source=source,
            row_id=row_id,
            subset_idx=subset_idx,
            metadata=_as_dict(request.get("metadata")),
        )
    except PromptConfigError as exc:
        raise WorkerContractError(str(exc)) from exc
    # Mirror the training format: append the SFT response marker so the model
    # sees the same "### Response:\n" trigger it learned to continue after.
    # Otherwise sampling can emit EOS as the first generated token.
    response_template = sft_response_template(prompts_cfg)
    return f"{prompt}\n{response_template}"


def _resolve_unsloth_runtime(request: Mapping[str, Any]) -> _UnslothRuntime:
    runtime_cfg = _as_dict(request.get("runtime_config"))
    inference_cfg = _as_dict(runtime_cfg.get("inference"))
    runtime = _as_dict(inference_cfg.get("runtime"))
    unsloth = _as_dict(runtime.get("unsloth"))
    return _UnslothRuntime(
        enabled=bool(unsloth.get("enabled", True)),
        fallback_to_transformers=bool(unsloth.get("fallback_to_transformers", True)),
    )


def _resolve_throughput_runtime(request: Mapping[str, Any]) -> _ThroughputRuntime:
    runtime_cfg = _as_dict(request.get("runtime_config"))
    inference_cfg = _as_dict(runtime_cfg.get("inference"))
    throughput_cfg = _as_dict(inference_cfg.get("throughput"))
    batching_cfg = _as_dict(throughput_cfg.get("batching"))

    strategy = str(batching_cfg.get("strategy", "token_budget")).strip() or "token_budget"
    raw_max_batch_tokens = batching_cfg.get("max_batch_tokens", 32768)
    if isinstance(raw_max_batch_tokens, bool) or not isinstance(raw_max_batch_tokens, int):
        max_batch_tokens = 32768
    else:
        max_batch_tokens = max(1024, raw_max_batch_tokens)

    raw_pad = batching_cfg.get("pad_to_multiple_of")
    pad_to_multiple_of: int | None = None
    if isinstance(raw_pad, int) and not isinstance(raw_pad, bool) and raw_pad > 0:
        pad_to_multiple_of = raw_pad

    return _ThroughputRuntime(
        strategy=strategy,
        max_batch_tokens=max_batch_tokens,
        pad_to_multiple_of=pad_to_multiple_of,
        preserve_order=bool(throughput_cfg.get("preserve_order", False)),
        restore_order_in_artifacts=bool(
            throughput_cfg.get("restore_order_in_artifacts", True)
        ),
    )


def _load_adapter_state_dict(adapter_path: Path) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    safetensors_files = sorted(adapter_path.glob("adapter_model*.safetensors"))
    if safetensors_files:
        from safetensors.torch import load_file

        for sf in safetensors_files:
            state_dict.update(load_file(str(sf), device="cpu"))
        return state_dict

    bin_file = adapter_path / "adapter_model.bin"
    if bin_file.exists():
        loaded = torch.load(str(bin_file), map_location="cpu")
        if not isinstance(loaded, dict):
            raise WorkerContractError(
                f"adapter_model.bin is not a dict at {bin_file}"
            )
        return loaded

    raise WorkerContractError(
        f"adapter weights not found under {adapter_path} "
        "(expected adapter_model*.safetensors or adapter_model.bin)"
    )


def _has_qwen35_language_model_prefix(model: Any) -> bool:
    for key, _ in model.named_parameters():
        if "language_model.layers." in key:
            return True
    return False


def _remap_qwen35_adapter_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], int]:
    remapped: dict[str, torch.Tensor] = {}
    changed = 0
    replacements = (
        ("model.layers.", "model.language_model.layers."),
        ("model.embed_tokens.", "model.language_model.embed_tokens."),
        ("model.norm.", "model.language_model.norm."),
    )
    for key, value in state_dict.items():
        new_key = key
        for src, dst in replacements:
            if src in new_key and dst not in new_key:
                new_key = new_key.replace(src, dst)
        if new_key != key:
            changed += 1
        remapped[new_key] = value
    return remapped, changed


def _attach_lora_adapter(
    model: Any,
    *,
    adapter_path: Path,
    adapter_name: str,
    is_trainable: bool,
) -> Any:
    if not _is_lora_adapter_path(adapter_path):
        raise WorkerContractError(
            f"adapter path is missing adapter_config.json: {adapter_path}"
        )

    try:
        from peft import PeftConfig, PeftModel
    except ModuleNotFoundError as exc:
        raise WorkerContractError("peft is required to load adapters for inference") from exc

    # We inspect adapter keys for debugging and (only when needed) apply a
    # Qwen3.5 prefix fix before loading.
    peft_config = PeftConfig.from_pretrained(str(adapter_path))
    state_dict = _load_adapter_state_dict(adapter_path)
    if state_dict:
        first_key = next(iter(state_dict.keys()))
        print(
            f"[inference-worker] adapter {adapter_name} first key: {first_key}",
            file=sys.stderr,
        )
    use_explicit_state_dict = False
    if _has_qwen35_language_model_prefix(model):
        old_layer_re = re.compile(r"(^|[._])model\\.layers\\.")
        old_prefix_count = sum(
            1
            for key in state_dict
            if old_layer_re.search(key) and "model.language_model.layers." not in key
        )
        new_prefix_count = sum(
            1 for key in state_dict if "model.language_model.layers." in key
        )
        print(
            f"[inference-worker] adapter {adapter_name} key-prefix counts: "
            f"old={old_prefix_count} new={new_prefix_count}",
            file=sys.stderr,
        )
        if old_prefix_count > 0:
            state_dict, changed = _remap_qwen35_adapter_state_dict(state_dict)
            print(
                f"[inference-worker] adapter key remap applied for {adapter_name}: "
                f"{changed} tensors",
                file=sys.stderr,
            )
            use_explicit_state_dict = changed > 0
    print(
        f"[inference-worker] adapter {adapter_name} load-mode: "
        f"{'state_dict' if use_explicit_state_dict else 'path'}",
        file=sys.stderr,
    )

    if hasattr(model, "load_adapter"):
        saved_checkpoint_mapping = None
        checkpoint_mapping_had_attr = hasattr(model, "_checkpoint_conversion_mapping")
        if checkpoint_mapping_had_attr:
            saved_checkpoint_mapping = getattr(model, "_checkpoint_conversion_mapping")
            try:
                mapping_size = (
                    len(saved_checkpoint_mapping)
                    if isinstance(saved_checkpoint_mapping, Mapping)
                    else None
                )
            except Exception:
                mapping_size = None
            print(
                f"[inference-worker] adapter {adapter_name} checkpoint-conversion mapping size: "
                f"{mapping_size if mapping_size is not None else 'unknown'}",
                file=sys.stderr,
            )
            # Prevent PEFT default conversion mapping from rewriting adapter keys.
            model._checkpoint_conversion_mapping = {}

        # Prefer path-based loading for compatibility. Use explicit state_dict
        # only when key remapping was required.
        if use_explicit_state_dict:
            load_attempts: list[dict[str, Any]] = [
                {
                    "peft_model_id": str(adapter_path),
                    "adapter_name": adapter_name,
                    "peft_config": peft_config,
                    "adapter_state_dict": state_dict,
                    "is_trainable": is_trainable,
                    "local_files_only": True,
                    "key_mapping": {},
                },
                {
                    "peft_model_id": str(adapter_path),
                    "adapter_name": adapter_name,
                    "peft_config": peft_config,
                    "adapter_state_dict": state_dict,
                    "is_trainable": is_trainable,
                    "key_mapping": {},
                },
            ]
        else:
            load_attempts = [
                {
                    "peft_model_id": str(adapter_path),
                    "adapter_name": adapter_name,
                    "is_trainable": is_trainable,
                    "local_files_only": True,
                    "key_mapping": {},
                },
                {
                    "peft_model_id": str(adapter_path),
                    "adapter_name": adapter_name,
                    "is_trainable": is_trainable,
                    "key_mapping": {},
                },
            ]

        last_exc: Exception | None = None
        try:
            for kwargs in load_attempts:
                try:
                    model.load_adapter(**kwargs)
                    return model
                except TypeError:
                    # Older signatures may reject optional kwargs; retry with
                    # progressively fewer optional arguments.
                    try:
                        fallback_kwargs = dict(kwargs)
                        fallback_kwargs.pop("local_files_only", None)
                        model.load_adapter(**fallback_kwargs)
                        return model
                    except TypeError:
                        try:
                            fallback_kwargs = dict(kwargs)
                            fallback_kwargs.pop("local_files_only", None)
                            fallback_kwargs.pop("key_mapping", None)
                            model.load_adapter(**fallback_kwargs)
                            return model
                        except Exception as exc:
                            last_exc = exc
                    except Exception as exc:
                        last_exc = exc
                except Exception as exc:
                    last_exc = exc

            if last_exc is not None:
                raise last_exc
        finally:
            if checkpoint_mapping_had_attr:
                model._checkpoint_conversion_mapping = saved_checkpoint_mapping
        return model

    # Fallback path for plain modules without mixin.
    return PeftModel.from_pretrained(
        model,
        str(adapter_path),
        adapter_name=adapter_name,
        is_trainable=is_trainable,
    )


def _load_model(request: Mapping[str, Any]) -> tuple[Any, Any]:
    runtime = _resolve_runtime(request)
    unsloth_runtime = _resolve_unsloth_runtime(request)
    if not unsloth_runtime.enabled:
        raise WorkerContractError(
            "unsloth-only inference requires inference.runtime.unsloth.enabled=true"
        )
    if unsloth_runtime.fallback_to_transformers:
        raise WorkerContractError(
            "unsloth-only inference requires inference.runtime.unsloth.fallback_to_transformers=false"
        )
    if not torch.cuda.is_available():
        raise WorkerContractError("unsloth-only inference requires CUDA GPU; torch.cuda.is_available()=false")

    try:
        from unsloth import FastLanguageModel
    except ModuleNotFoundError as exc:
        raise WorkerContractError("unsloth package is required for unsloth-only inference") from exc

    q_tag = str(request.get("q_tag", "q1"))
    collapse_path: Path | None = None
    collapse_noop = False
    use_q2_adapter_dir_model_ref = False
    if q_tag == "q2":
        collapse_adapter = request.get("collapse_adapter")
        if not isinstance(collapse_adapter, str) or not collapse_adapter.strip():
            raise WorkerContractError("infer-q2 requires non-empty collapse_adapter path")
        collapse_path = Path(collapse_adapter)
        collapse_noop = _is_noop_collapse_adapter_path(collapse_path)
        if not _is_lora_adapter_path(collapse_path):
            if not collapse_noop:
                raise WorkerContractError(
                    "collapse adapter is missing adapter_config.json; "
                    "run train-collapse-lora first"
                )
            print(
                f"[inference-worker] q2 collapse adapter is no-op; using base model only ({collapse_path})",
                file=sys.stderr,
            )
        else:
            # Notebook parity path:
            # load the collapse LoRA directory directly as model_name so Unsloth/PEFT
            # restores the adapter stack in one step, avoiding hot-load key remap drift.
            use_q2_adapter_dir_model_ref = True
            print(
                f"[inference-worker] q2 load strategy: direct-adapter-model-ref ({collapse_path})",
                file=sys.stderr,
            )

    model_name_for_load = str(collapse_path) if use_q2_adapter_dir_model_ref else runtime.model_ref
    load_in_4bit = bool(
        _as_dict(_as_dict(request.get("runtime_config")).get("model")).get("load_in_4bit", False)
    )
    kwargs: dict[str, Any] = {
        "model_name": model_name_for_load,
        "max_seq_length": runtime.max_seq_length,
        "load_in_4bit": load_in_4bit,
    }
    if runtime.torch_dtype is not None and runtime.torch_dtype != "auto":
        kwargs["dtype"] = runtime.torch_dtype
    if runtime.trust_remote_code:
        kwargs["trust_remote_code"] = True

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)
    except TypeError:
        kwargs.pop("trust_remote_code", None)
        model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)
    except Exception as exc:
        raise WorkerContractError(f"unsloth loading failed: {exc}") from exc

    base_checkpoint = request.get("base_checkpoint")
    base_update_loaded = False
    if (not use_q2_adapter_dir_model_ref) and isinstance(base_checkpoint, str) and base_checkpoint.strip():
        cp = Path(base_checkpoint)
        if _is_lora_adapter_path(cp):
            model = _attach_lora_adapter(
                model,
                adapter_path=cp,
                adapter_name="base_update",
                is_trainable=False,
            )
            model.set_adapter("base_update")
            base_update_loaded = True

    if q_tag == "q2":
        if use_q2_adapter_dir_model_ref:
            # Adapter already loaded via from_pretrained(model_name=<collapse_adapter_dir>).
            if hasattr(model, "active_adapters"):
                try:
                    active = model.active_adapters()
                    print(
                        f"[inference-worker] q2 direct-adapter active_adapters={active}",
                        file=sys.stderr,
                    )
                except Exception:
                    pass
        elif collapse_noop:
            # No collapse adapter weights for this subset (all samples filtered by length).
            # Continue with base model so the pipeline can progress without hard failure.
            pass
        else:
            assert collapse_path is not None
            model = _attach_lora_adapter(
                model,
                adapter_path=collapse_path,
                adapter_name="collapse_probe",
                is_trainable=False,
            )

            if hasattr(model, "set_adapter"):
                if base_update_loaded:
                    try:
                        model.set_adapter(["base_update", "collapse_probe"])
                    except Exception:
                        model.set_adapter("collapse_probe")
                        print(
                            "[inference-worker] q2: failed to co-activate "
                            "base_update+collapse_probe; using collapse_probe only",
                            file=sys.stderr,
                        )
                else:
                    model.set_adapter("collapse_probe")

    # Activate Unsloth inference path after all adapters are attached.
    # Calling this earlier can change module naming/layout and break adapter key matching.
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = runtime.padding_side
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    return model, tokenizer


def _decoding_config(request: Mapping[str, Any]) -> tuple[int, bool, float, float | None]:
    decoding_cfg = _as_dict(request.get("decoding"))
    max_new_tokens = int(decoding_cfg.get("max_new_tokens", 256) or 256)
    if max_new_tokens <= 0:
        max_new_tokens = 256
    do_sample = bool(decoding_cfg.get("do_sample", False))
    temperature = float(decoding_cfg.get("temperature", 0.0) or 0.0)
    top_p_raw = decoding_cfg.get("top_p", None)
    top_p = float(top_p_raw) if isinstance(top_p_raw, (int, float)) else None
    return max_new_tokens, do_sample, temperature, top_p


def _request_prompt(request: Mapping[str, Any]) -> str:
    source = str(request.get("source", "")).strip()
    if not source:
        raise WorkerContractError("inference request row missing source text")
    return _build_prompt(source, request=request)


def _build_batch_items(
    *,
    requests: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    throughput: _ThroughputRuntime,
) -> list[_BatchItem]:
    prompts = [_request_prompt(row) for row in requests]
    tokenized = tokenizer(prompts, add_special_tokens=True, return_length=True, truncation=False)
    lengths = tokenized.get("length")
    if not isinstance(lengths, list) or len(lengths) != len(requests):
        raise WorkerContractError("failed to compute prompt lengths for batching")
    items = [
        _BatchItem(
            order_idx=idx,
            request=requests[idx],
            prompt=prompts[idx],
            prompt_tokens=int(lengths[idx]),
        )
        for idx in range(len(requests))
    ]
    if throughput.preserve_order:
        return items
    return sorted(items, key=lambda item: item.prompt_tokens)


def _estimate_batch_tokens(max_prompt_tokens: int, rows: int, max_new_tokens: int) -> int:
    return (max_prompt_tokens + max_new_tokens) * rows


def _group_batches(
    *,
    items: Sequence[_BatchItem],
    max_batch_tokens: int,
) -> list[list[_BatchItem]]:
    if not items:
        return []
    out: list[list[_BatchItem]] = []
    cursor: list[_BatchItem] = []
    max_prompt = 0
    max_new_tokens = 0
    for item in items:
        item_new_tokens, _, _, _ = _decoding_config(item.request)
        next_max_prompt = max(max_prompt, item.prompt_tokens)
        next_max_new = max(max_new_tokens, item_new_tokens)
        next_rows = len(cursor) + 1
        estimated = _estimate_batch_tokens(next_max_prompt, next_rows, next_max_new)
        if cursor and estimated > max_batch_tokens:
            out.append(cursor)
            cursor = []
            max_prompt = 0
            max_new_tokens = 0
        cursor.append(item)
        max_prompt = max(max_prompt, item.prompt_tokens)
        max_new_tokens = max(max_new_tokens, item_new_tokens)
    if cursor:
        out.append(cursor)
    return out


def _generate_batch(
    *,
    model: Any,
    tokenizer: Any,
    batch: Sequence[_BatchItem],
    pad_to_multiple_of: int | None,
) -> dict[int, str]:
    prompts = [item.prompt for item in batch]
    max_new_tokens = max(_decoding_config(item.request)[0] for item in batch)
    any_do_sample = any(_decoding_config(item.request)[1] for item in batch)
    max_temperature = max(_decoding_config(item.request)[2] for item in batch)
    valid_top_ps = [
        top_p
        for (_, _, _, top_p) in (_decoding_config(item.request) for item in batch)
        if isinstance(top_p, float)
    ]
    top_p = min(valid_top_ps) if valid_top_ps else None

    tokenized_kwargs: dict[str, Any] = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": False,
    }
    if pad_to_multiple_of is not None:
        tokenized_kwargs["pad_to_multiple_of"] = pad_to_multiple_of

    inputs = tokenizer(prompts, **tokenized_kwargs)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    input_lengths = attention_mask.sum(dim=1).tolist()

    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": any_do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if any_do_sample and max_temperature > 0:
        generate_kwargs["temperature"] = max_temperature
        if top_p is not None:
            generate_kwargs["top_p"] = top_p
    else:
        generate_kwargs["temperature"] = 0.0

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)

    resolved: dict[int, str] = {}
    for idx, item in enumerate(batch):
        prompt_len = int(input_lengths[idx])
        text = tokenizer.decode(output_ids[idx][prompt_len:], skip_special_tokens=True).strip()
        if not text:
            raise WorkerContractError(
                f"inference generation returned empty translation for id={item.request.get('id')}"
            )
        resolved[item.order_idx] = text
    return resolved


def _generate_responses(
    *,
    model: Any,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    throughput: _ThroughputRuntime,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = _build_batch_items(requests=requests, tokenizer=tokenizer, throughput=throughput)
    batches = _group_batches(items=items, max_batch_tokens=throughput.max_batch_tokens)
    resolved_mt: dict[int, str] = {}
    oom_retry_count = 0

    total_rows = len(items)
    done_rows = 0
    print(
        f"[inference-worker] starting: {total_rows} rows, {len(batches)} batches",
        file=sys.stderr,
    )

    for batch_idx, batch in enumerate(batches):
        queue: list[list[_BatchItem]] = [list(batch)]
        while queue:
            chunk = queue.pop(0)
            try:
                generated = _generate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    batch=chunk,
                    pad_to_multiple_of=throughput.pad_to_multiple_of,
                )
                resolved_mt.update(generated)
            except torch.cuda.OutOfMemoryError:
                oom_retry_count += 1
                if len(chunk) <= 1:
                    item = chunk[0]
                    req_id = str(item.request.get("id", ""))
                    raise WorkerContractError(
                        f"OOM on single-row inference request id={req_id}; lower max_new_tokens or batch token budget"
                    )
                midpoint = int(math.ceil(len(chunk) / 2))
                queue.insert(0, chunk[midpoint:])
                queue.insert(0, chunk[:midpoint])
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        done_rows += len(batch)
        print(
            f"[inference-worker] batch {batch_idx + 1}/{len(batches)} done ({done_rows}/{total_rows} rows)",
            file=sys.stderr,
        )

    ordered_items = items if throughput.restore_order_in_artifacts else list(items)
    if throughput.restore_order_in_artifacts:
        ordered_items = sorted(items, key=lambda item: item.order_idx)

    responses: list[dict[str, Any]] = []
    for item in ordered_items:
        req_id = str(item.request.get("id", ""))
        order_idx = int(item.order_idx)
        mt = resolved_mt.get(item.order_idx)
        if mt is None:
            responses.append(
                {
                    "id": req_id,
                    "order_idx": order_idx,
                    "status": "failed",
                    "mt": "",
                    "error": "missing generated text",
                }
            )
            continue
        responses.append(
            {"id": req_id, "order_idx": order_idx, "status": "ok", "mt": mt, "error": None}
        )

    metrics = {
        "batch_count": len(batches),
        "avg_batch_rows": round(sum(len(batch) for batch in batches) / max(len(batches), 1), 3),
        "avg_batch_tokens": round(
            sum(
                _estimate_batch_tokens(
                    max(item.prompt_tokens for item in batch),
                    len(batch),
                    max(_decoding_config(item.request)[0] for item in batch),
                )
                for batch in batches
            )
            / max(len(batches), 1),
            3,
        ),
        "oom_retry_count": oom_retry_count,
    }
    return responses, metrics


def main(argv: list[str] | None = None) -> int:
    args = parse_worker_args(description="Real inference worker", argv=argv)

    requests = read_jsonl(args.input_path)
    schema = validate_phase_request_rows(requests, args=args, context="inference")
    if not requests:
        write_jsonl(args.output_path, [], ensure_ascii=False)
        return 0

    throughput = _resolve_throughput_runtime(requests[0])
    try:
        model, tokenizer = _load_model(requests[0])
        responses, metrics = _generate_responses(
            model=model,
            tokenizer=tokenizer,
            requests=requests,
            throughput=throughput,
        )
        print(
            "[inference-worker] "
            f"batch_count={metrics['batch_count']} "
            f"avg_batch_rows={metrics['avg_batch_rows']} "
            f"avg_batch_tokens={metrics['avg_batch_tokens']} "
            f"oom_retry_count={metrics['oom_retry_count']}",
            file=sys.stderr,
        )
    except Exception as exc:
        responses = [
            {
                "id": str(row.get("id", "")),
                "order_idx": int(row.get("order_idx", idx)),
                "status": "failed",
                "mt": "",
                "error": str(exc),
            }
            for idx, row in enumerate(requests)
        ]

    validate_phase_response_rows(responses, schema=schema, context="inference")
    write_jsonl(args.output_path, responses, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
