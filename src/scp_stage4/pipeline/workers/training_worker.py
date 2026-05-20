"""Real training worker for subprocess runtime.

Implements:
- train-collapse-lora
- unload-collapse-lora
- update-base

Training backend is Unsloth + TRL SFTTrainer as required by AGENTS.md.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from scp_stage4.data import read_jsonl, write_jsonl
from scp_stage4.pipeline.prompting import (
    PromptConfigError,
    render_sft_text,
    sft_response_template,
)
from scp_stage4.pipeline.workers.common import (
    WorkerContractError,
    parse_worker_args,
    validate_phase_request_rows,
    validate_phase_response_rows,
)

_DEFAULT_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
_NOOP_COLLAPSE_MARKER = "NOOP_COLLAPSE.json"


@dataclass(frozen=True)
class _TrainRuntime:
    model_ref: str
    max_seq_length: int
    load_in_4bit: bool
    dtype: Any
    attention_impl: str | None
    trust_remote_code: bool


@dataclass(frozen=True)
class _DDPRuntime:
    enabled: bool
    local_rank: int
    rank: int
    world_size: int
    is_main_process: bool


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value if value > 0 else default


def _dtype_from_config(dtype_value: Any) -> Any:
    if dtype_value is None:
        return None
    text = str(dtype_value).strip().lower()
    try:
        import torch
    except ModuleNotFoundError:
        return None
    if text in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if text in {"fp16", "float16"}:
        return torch.float16
    if text in {"fp32", "float32"}:
        return torch.float32
    return None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _resolve_ddp_runtime() -> _DDPRuntime:
    world_size = max(1, _env_int("WORLD_SIZE", 1))
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    enabled = world_size > 1 or ("LOCAL_RANK" in os.environ and "RANK" in os.environ)
    if not enabled:
        return _DDPRuntime(
            enabled=False,
            local_rank=0,
            rank=0,
            world_size=1,
            is_main_process=True,
        )
    return _DDPRuntime(
        enabled=True,
        local_rank=max(0, local_rank),
        rank=max(0, rank),
        world_size=max(1, world_size),
        is_main_process=(rank == 0),
    )


def _init_ddp_runtime() -> _DDPRuntime:
    ddp = _resolve_ddp_runtime()
    if not ddp.enabled:
        return ddp

    try:
        import torch
        import torch.distributed as dist
    except ModuleNotFoundError as exc:
        raise WorkerContractError("torch distributed is required for DDP training runtime") from exc

    if not dist.is_available():
        raise WorkerContractError("torch.distributed is not available for DDP runtime")

    if torch.cuda.is_available():
        torch.cuda.set_device(ddp.local_rank)
        backend = "nccl"
    else:
        backend = "gloo"

    if not dist.is_initialized():
        import datetime
        init_kwargs: dict[str, Any] = {
            "backend": backend,
            "init_method": "env://",
            "rank": ddp.rank,
            "world_size": ddp.world_size,
            "timeout": datetime.timedelta(minutes=10),
        }
        if backend == "nccl" and torch.cuda.is_available():
            # Pin device explicitly so NCCL does not guess from global rank;
            # heterogeneous rank->GPU maps would otherwise hang collectives.
            init_kwargs["device_id"] = torch.device(f"cuda:{ddp.local_rank}")
        dist.init_process_group(**init_kwargs)
    return ddp


def _ddp_barrier(ddp: _DDPRuntime) -> None:
    if not ddp.enabled:
        return
    try:
        import torch.distributed as dist
    except ModuleNotFoundError:
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _shutdown_ddp_runtime(ddp: _DDPRuntime) -> None:
    if not ddp.enabled:
        return
    try:
        import torch.distributed as dist
    except ModuleNotFoundError:
        return
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _resolve_train_runtime(row: Mapping[str, Any]) -> _TrainRuntime:
    model_cfg = _as_dict(row.get("model"))
    model_name = str(model_cfg.get("name", "")).strip()
    if not model_name:
        raise WorkerContractError("training request row missing model.name")

    max_seq_length = model_cfg.get("max_seq_length")
    if isinstance(max_seq_length, bool) or not isinstance(max_seq_length, int) or max_seq_length <= 0:
        max_length = model_cfg.get("max_length")
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
            max_seq_length = 2048
        else:
            max_seq_length = max_length

    base_checkpoint = row.get("base_checkpoint")
    model_ref = model_name
    if isinstance(base_checkpoint, str) and base_checkpoint.strip():
        checkpoint_path = Path(base_checkpoint)
        if checkpoint_path.exists():
            model_ref = base_checkpoint
        elif bool(row.get("requires_base_checkpoint", False)):
            raise WorkerContractError(f"required base_checkpoint path not found: {checkpoint_path}")
    elif bool(row.get("requires_base_checkpoint", False)):
        raise WorkerContractError("required base_checkpoint is missing")

    return _TrainRuntime(
        model_ref=model_ref,
        max_seq_length=int(max_seq_length),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
        dtype=_dtype_from_config(model_cfg.get("dtype")),
        attention_impl=_resolve_attention_impl_from_config(model_cfg),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )


def _resolve_attention_impl_from_config(model_cfg: Mapping[str, Any]) -> str | None:
    raw = model_cfg.get("attention_impl")
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def _resolve_attention_impl(runtime: _TrainRuntime) -> str:
    env_value = os.environ.get("ATTN_IMPLEMENTATION")
    if env_value is not None:
        env_value = env_value.strip()
        if env_value:
            return env_value
    if runtime.attention_impl:
        return runtime.attention_impl
    return "sdpa"


def _format_sft_text(
    source: str,
    target: str,
    *,
    prompts_cfg: Mapping[str, Any] | None,
) -> str:
    try:
        return render_sft_text(
            prompts=prompts_cfg,
            source=source,
            target=target,
        )
    except PromptConfigError as exc:
        raise WorkerContractError(str(exc)) from exc


def _build_dataset(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    prompts_cfg: Mapping[str, Any] | None,
    max_seq_length: int,
) -> Any:
    try:
        from datasets import Dataset
    except ModuleNotFoundError as exc:
        raise WorkerContractError("datasets package is required for training worker") from exc

    eos = getattr(tokenizer, "eos_token", None) or ""
    payload = []
    row_ids: list[str] = []
    for row in rows:
        source = str(row.get("source", "")).strip()
        target = str(row.get("target", "")).strip()
        if not source or not target:
            continue
        row_id = str(row.get("id", "")).strip() or f"row_{len(payload):06d}"
        payload.append(
            {
                "text": _format_sft_text(
                    source,
                    target,
                    prompts_cfg=prompts_cfg,
                )
                + (eos if eos else ""),
            }
        )
        row_ids.append(row_id)
    if not payload:
        raise WorkerContractError("no valid source/target rows for training")
    keep_indices, over_limit = _filter_training_text_indices_by_length(
        tokenizer=tokenizer,
        texts=[str(item["text"]) for item in payload],
        row_ids=row_ids,
        max_seq_length=max_seq_length,
    )
    if over_limit:
        sample = ", ".join(f"{row_id}:{length}" for row_id, length in over_limit[:5])
        print(
            "[training-worker] filtered over-limit samples before SFT; "
            f"max_seq_length={max_seq_length}, "
            f"filtered_rows={len(over_limit)}/{len(payload)}, "
            f"examples=[{sample}]",
            file=sys.stderr,
            flush=True,
        )
    payload = [payload[idx] for idx in keep_indices]
    if not payload:
        raise WorkerContractError(
            "all training samples exceed model context window after filtering; "
            f"max_seq_length={max_seq_length}, filtered_rows={len(over_limit)}"
        )
    return Dataset.from_list(payload)


def _tokenized_text_lengths(tokenizer: Any, texts: list[str]) -> list[int]:
    try:
        tokenized = tokenizer(
            texts,
            add_special_tokens=True,
            return_length=True,
            truncation=False,
        )
    except Exception as exc:
        raise WorkerContractError(f"failed to tokenize training texts for length audit: {exc}") from exc

    lengths = tokenized.get("length") if isinstance(tokenized, Mapping) else None
    if isinstance(lengths, list) and len(lengths) == len(texts):
        out: list[int] = []
        for idx, value in enumerate(lengths):
            try:
                parsed = int(value)
            except Exception as exc:
                raise WorkerContractError(
                    f"tokenizer length value is not numeric at index={idx}: {value!r}"
                ) from exc
            out.append(parsed)
        return out

    out = []
    for idx, text in enumerate(texts):
        try:
            encoded = tokenizer.encode(text, add_special_tokens=True)
        except Exception as exc:
            raise WorkerContractError(
                f"failed to encode training text for length audit at index={idx}: {exc}"
            ) from exc
        out.append(len(encoded))
    return out


def _filter_training_text_indices_by_length(
    *,
    tokenizer: Any,
    texts: list[str],
    row_ids: list[str],
    max_seq_length: int,
) -> tuple[list[int], list[tuple[str, int]]]:
    if max_seq_length <= 0:
        raise WorkerContractError(f"max_seq_length must be positive, got={max_seq_length}")
    if len(texts) != len(row_ids):
        raise WorkerContractError(
            f"length audit mismatch: texts={len(texts)} row_ids={len(row_ids)}"
        )
    if not texts:
        return [], []

    keep_indices: list[int] = []
    over_limit: list[tuple[str, int]] = []
    batch_size = 1024
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        lengths = _tokenized_text_lengths(tokenizer, chunk)
        if len(lengths) != len(chunk):
            raise WorkerContractError(
                f"tokenizer length audit returned mismatched count: expected={len(chunk)} got={len(lengths)}"
            )
        for offset, length in enumerate(lengths):
            idx = start + offset
            if length > max_seq_length:
                over_limit.append((row_ids[idx], int(length)))
                continue
            keep_indices.append(idx)
    return keep_indices, over_limit


def _is_context_overflow_skip_error(exc: WorkerContractError) -> bool:
    message = str(exc)
    return (
        "all training samples exceed model context window after filtering" in message
        or "no valid source/target rows for training" in message
    )


def _write_noop_collapse_marker(adapter_path: Path, *, reason: str) -> None:
    marker_path = adapter_path / _NOOP_COLLAPSE_MARKER
    payload = {
        "status": "ok",
        "no_op": True,
        "reason": reason,
    }
    marker_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _is_noop_collapse_adapter_path(path: Path) -> bool:
    return path.exists() and (path / _NOOP_COLLAPSE_MARKER).exists()


def _use_unsloth() -> bool:
    return os.environ.get("DISABLE_UNSLOTH", "").strip() not in ("1", "true", "yes")


def _load_hf_model(runtime: _TrainRuntime) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ModuleNotFoundError as exc:
        raise WorkerContractError("transformers/torch required for HF training") from exc

    kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": runtime.model_ref,
        "trust_remote_code": runtime.trust_remote_code,
    }
    if runtime.dtype is not None:
        kwargs["torch_dtype"] = runtime.dtype
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if runtime.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=kwargs.get("torch_dtype", torch.bfloat16),
            )
        except ImportError:
            pass

    model = AutoModelForCausalLM.from_pretrained(**kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        runtime.model_ref,
        trust_remote_code=runtime.trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _load_unsloth_model(
    runtime: _TrainRuntime, *, full_finetuning: bool = False
) -> tuple[Any, Any]:
    if not _use_unsloth():
        return _load_hf_model(runtime)

    try:
        import inspect

        from unsloth import FastLanguageModel, FastVisionModel
    except ModuleNotFoundError as exc:
        raise WorkerContractError("unsloth package is required for training runtime") from exc

    kwargs: dict[str, Any] = {
        "model_name": runtime.model_ref,
        "max_seq_length": runtime.max_seq_length,
        "load_in_4bit": runtime.load_in_4bit,
    }
    if runtime.dtype is not None:
        kwargs["dtype"] = runtime.dtype
    if runtime.trust_remote_code:
        kwargs["trust_remote_code"] = True

    attn_impl = _resolve_attention_impl(runtime)
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    model = tokenizer = None
    load_errors: list[str] = []
    for candidate_cls in (FastVisionModel, FastLanguageModel):
        attempt_kwargs = dict(kwargs)
        # Tell Unsloth to keep the full model trainable. Without this Unsloth
        # silently switches to "16bit LoRA" mode, so for mode=full_weight the
        # base weights never receive gradients and the saved full_weight_model
        # is identical to the original base checkpoint.
        if (
            full_finetuning
            and "full_finetuning"
            in inspect.signature(candidate_cls.from_pretrained).parameters
        ):
            attempt_kwargs["full_finetuning"] = True
        try:
            model, tokenizer = candidate_cls.from_pretrained(**attempt_kwargs)
            break
        except TypeError:
            attempt_kwargs.pop("trust_remote_code", None)
            attempt_kwargs.pop("attn_implementation", None)
            try:
                model, tokenizer = candidate_cls.from_pretrained(**attempt_kwargs)
                break
            except Exception as exc:
                load_errors.append(f"{candidate_cls.__name__}: {type(exc).__name__}: {exc}")
        except Exception as exc:
            load_errors.append(f"{candidate_cls.__name__}: {type(exc).__name__}: {exc}")
    if model is None or tokenizer is None:
        raise WorkerContractError(
            "Failed to load model via Unsloth (Vision then Language).\n"
            + "\n".join(load_errors)
        )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _model_has_peft(model: Any) -> bool:
    return hasattr(model, "peft_config")


def _ensure_lora_model_hf(
    *,
    model: Any,
    lora_cfg: Mapping[str, Any],
) -> Any:
    try:
        from peft import LoraConfig, get_peft_model
    except ModuleNotFoundError as exc:
        raise WorkerContractError("peft package is required for HF LoRA setup") from exc

    rank = _as_positive_int(lora_cfg.get("rank"), 8)
    alpha = _as_positive_int(lora_cfg.get("alpha"), rank * 2)
    dropout = float(lora_cfg.get("dropout", 0.0) or 0.0)
    bias = str(lora_cfg.get("bias", "none"))
    target_modules_raw = lora_cfg.get("target_modules")
    if isinstance(target_modules_raw, list) and target_modules_raw:
        target_modules = [str(m) for m in target_modules_raw if str(m).strip()]
    else:
        target_modules = list(_DEFAULT_LORA_TARGETS)

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def _ensure_lora_model(
    *,
    model: Any,
    runtime: _TrainRuntime,
    lora_cfg: Mapping[str, Any],
    seed: int,
) -> Any:
    if _model_has_peft(model):
        return model

    if not _use_unsloth():
        return _ensure_lora_model_hf(model=model, lora_cfg=lora_cfg)

    try:
        from unsloth import FastLanguageModel
    except ModuleNotFoundError as exc:
        raise WorkerContractError("unsloth package is required for LoRA setup") from exc

    rank = _as_positive_int(lora_cfg.get("rank"), 8)
    alpha = _as_positive_int(lora_cfg.get("alpha"), rank * 2)
    dropout = float(lora_cfg.get("dropout", 0.0) or 0.0)
    bias = str(lora_cfg.get("bias", "none"))
    target_modules_raw = lora_cfg.get("target_modules")
    target_modules: list[str] | str
    if isinstance(target_modules_raw, str) and target_modules_raw.strip():
        target_modules = target_modules_raw.strip()
    elif isinstance(target_modules_raw, list) and target_modules_raw:
        parsed = [str(module) for module in target_modules_raw if str(module).strip()]
        target_modules = parsed if parsed else list(_DEFAULT_LORA_TARGETS)
    else:
        target_modules = list(_DEFAULT_LORA_TARGETS)

    use_rslora = bool(lora_cfg.get("use_rslora", False))
    loftq_config = lora_cfg.get("loftq_config", None)

    peft_kwargs: dict[str, Any] = {
        "r": rank,
        "target_modules": target_modules,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "bias": bias,
        "use_gradient_checkpointing": "unsloth",
        "random_state": int(seed),
        "max_seq_length": runtime.max_seq_length,
        "use_rslora": use_rslora,
        "loftq_config": loftq_config,
    }
    supported = inspect.signature(FastLanguageModel.get_peft_model).parameters
    filtered_kwargs = {key: value for key, value in peft_kwargs.items() if key in supported}
    return FastLanguageModel.get_peft_model(model, **filtered_kwargs)


def _trainer_device_flags() -> tuple[bool, bool]:
    try:
        import torch
    except ModuleNotFoundError:
        return False, False

    if not torch.cuda.is_available():
        return False, False
    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    return (not bf16_supported), bf16_supported


def _resolve_wandb_report_to(
    logging_cfg: Mapping[str, Any],
    *,
    is_main_process: bool,
) -> list[str]:
    report_to_raw = logging_cfg.get("report_to", [])
    report_to: list[str] = list(report_to_raw) if isinstance(report_to_raw, (list, tuple)) else []
    if not is_main_process:
        return [target for target in report_to if target != "wandb"]
    wandb_cfg = _as_dict(logging_cfg.get("wandb", {}))
    if "wandb" in report_to and wandb_cfg.get("enabled", True):
        try:
            import wandb  # type: ignore[import-untyped]

            init_kwargs: dict[str, Any] = {"project": str(wandb_cfg.get("project", "scp_main"))}
            entity = wandb_cfg.get("entity")
            if entity:
                init_kwargs["entity"] = str(entity)
            tags = wandb_cfg.get("tags")
            resolved_tags: list[str] = []
            if isinstance(tags, list):
                resolved_tags.extend(str(t) for t in tags if str(t).strip())
            phase_value = str(logging_cfg.get("phase", "")).strip()
            if phase_value:
                # Keep both detailed and coarse phase tags for filtering.
                phase_tag = f"phase:{phase_value}"
                if phase_tag not in resolved_tags:
                    resolved_tags.append(phase_tag)
                coarse_tag = None
                if phase_value == "train-collapse-lora":
                    coarse_tag = "collapse"
                elif phase_value == "update-base":
                    coarse_tag = "update"
                if coarse_tag and coarse_tag not in resolved_tags:
                    resolved_tags.append(coarse_tag)
            if resolved_tags:
                init_kwargs["tags"] = resolved_tags
            notes = wandb_cfg.get("notes", "")
            if notes:
                init_kwargs["notes"] = str(notes)
            run_name = logging_cfg.get("run_name")
            if run_name:
                init_kwargs["name"] = str(run_name)
            wandb.init(**init_kwargs)
        except Exception:
            report_to = [r for r in report_to if r != "wandb"]
    return report_to


def _resolve_response_template(train_cfg: Mapping[str, Any], *, phase: str) -> str:
    runtime_prompts = _as_dict(_as_dict(train_cfg).get("runtime_prompts"))
    if runtime_prompts:
        try:
            return sft_response_template(runtime_prompts)
        except PromptConfigError as exc:
            raise WorkerContractError(str(exc)) from exc

    batching_cfg = _as_dict(train_cfg.get("batching"))
    candidate = batching_cfg.get("response_template")
    if not isinstance(candidate, str) or not candidate.strip():
        candidate = train_cfg.get("response_template")
    if not isinstance(candidate, str) or not candidate.strip():
        raise WorkerContractError(
            f"{phase} requires non-empty response_template config "
            "(prompts.sft.response_template or training.<phase>.batching.response_template)"
        )
    return candidate


def _build_response_only_collator(tokenizer: Any, *, response_template: str) -> Any:
    try:
        from trl import DataCollatorForCompletionOnlyLM

        return DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=tokenizer,
        )
    except (ImportError, Exception):
        pass

    import torch
    from dataclasses import dataclass

    template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    if not template_ids:
        raise WorkerContractError(
            f"response_template {response_template!r} tokenizes to empty sequence"
        )

    @dataclass
    class _ResponseOnlyCollator:
        tokenizer: Any
        _template_ids: list[int] = None

        def __post_init__(self) -> None:
            self._template_ids = template_ids

        def __call__(self, features: list[dict]) -> dict:
            batch = self.tokenizer.pad(
                features,
                padding=True,
                return_tensors="pt",
            )
            labels = batch["input_ids"].clone()
            for i in range(labels.size(0)):
                ids = batch["input_ids"][i].tolist()
                mask_until = 0
                tpl = self._template_ids
                for j in range(len(ids) - len(tpl) + 1):
                    if ids[j : j + len(tpl)] == tpl:
                        mask_until = j + len(tpl)
                labels[i, :mask_until] = -100
                if "attention_mask" in batch:
                    pad_mask = batch["attention_mask"][i] == 0
                    labels[i][pad_mask] = -100
            batch["labels"] = labels
            return batch

    return _ResponseOnlyCollator(tokenizer=tokenizer)


def _instantiate_trainer(
    *,
    model: Any,
    tokenizer: Any,
    dataset: Any,
    output_dir: Path,
    train_cfg: Mapping[str, Any],
    max_seq_length: int,
    response_template: str,
    logging_cfg: Mapping[str, Any] | None = None,
    is_main_process: bool = True,
) -> Any:
    try:
        from trl import SFTTrainer
    except ModuleNotFoundError as exc:
        raise WorkerContractError("trl package is required for SFT training") from exc

    collator = _build_response_only_collator(
        tokenizer,
        response_template=response_template,
    )
    init_sig = inspect.signature(SFTTrainer.__init__)
    if "data_collator" not in init_sig.parameters:
        raise WorkerContractError(
            "installed TRL SFTTrainer does not accept data_collator; "
            "response-only masking cannot be guaranteed"
        )

    report_to = _resolve_wandb_report_to(
        logging_cfg or {},
        is_main_process=is_main_process,
    )

    fp16, bf16 = _trainer_device_flags()
    common_train_args: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": _env_int("TRAIN_BATCH_SIZE", 0) or _as_positive_int(
            _as_dict(train_cfg.get("batching")).get("per_device_train_batch_size", train_cfg.get("per_device_train_batch_size")),
            1,
        ),
        "gradient_accumulation_steps": _as_positive_int(
            _as_dict(train_cfg.get("batching")).get("gradient_accumulation_steps", train_cfg.get("gradient_accumulation_steps")),
            1,
        ),
        "num_train_epochs": float(train_cfg.get("num_train_epochs", 1.0) or 1.0),
        "learning_rate": float(
            _as_dict(train_cfg.get("optimizer")).get("learning_rate", train_cfg.get("learning_rate", 2e-5))
            or 2e-5
        ),
        "weight_decay": float(_as_dict(train_cfg.get("optimizer")).get("weight_decay", 0.0) or 0.0),
        "warmup_ratio": float(_as_dict(train_cfg.get("optimizer")).get("warmup_ratio", 0.0) or 0.0),
        "lr_scheduler_type": str(
            _as_dict(train_cfg.get("optimizer")).get("lr_scheduler_type", "cosine")
        ),
        "optim": str(_as_dict(train_cfg.get("optimizer")).get("optim", "adamw_torch")),
        "max_grad_norm": float(_as_dict(train_cfg.get("optimizer")).get("max_grad_norm", 1.0) or 1.0),
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": report_to,
        "fp16": fp16,
        "bf16": bf16,
        "ddp_find_unused_parameters": False,
    }

    trainer = None
    trainer_errors: list[str] = []

    # Try TRL SFTConfig path first.
    try:
        from trl import SFTConfig

        sft_args = dict(common_train_args)
        # TRL v0.2x uses max_length, older versions use max_seq_length.
        sft_args["max_length"] = max_seq_length
        sft_args["dataset_text_field"] = "text"
        sft_args["packing"] = bool(_as_dict(train_cfg.get("batching")).get("packing", False))

        try:
            args_obj = SFTConfig(**sft_args)
        except TypeError:
            sft_args.pop("max_length", None)
            sft_args["max_seq_length"] = max_seq_length
            args_obj = SFTConfig(**sft_args)

        kwargs: dict[str, Any] = {
            "model": model,
            "train_dataset": dataset,
            "args": args_obj,
        }
        if "processing_class" in init_sig.parameters:
            kwargs["processing_class"] = tokenizer
        elif "tokenizer" in init_sig.parameters:
            kwargs["tokenizer"] = tokenizer
        kwargs["data_collator"] = collator
        trainer = SFTTrainer(**kwargs)
    except Exception as exc:
        trainer_errors.append(f"SFTConfig path failed: {exc}")

    if trainer is not None:
        return trainer

    # Fallback for older TRL signatures using transformers.TrainingArguments.
    try:
        from transformers import TrainingArguments

        args_obj = TrainingArguments(**common_train_args)
        kwargs = {
            "model": model,
            "train_dataset": dataset,
            "args": args_obj,
        }
        if "processing_class" in init_sig.parameters:
            kwargs["processing_class"] = tokenizer
        elif "tokenizer" in init_sig.parameters:
            kwargs["tokenizer"] = tokenizer
        kwargs["data_collator"] = collator
        if "dataset_text_field" in init_sig.parameters:
            kwargs["dataset_text_field"] = "text"
        if "max_seq_length" in init_sig.parameters:
            kwargs["max_seq_length"] = max_seq_length
        if "packing" in init_sig.parameters:
            kwargs["packing"] = bool(_as_dict(train_cfg.get("batching")).get("packing", False))
        return SFTTrainer(**kwargs)
    except Exception as exc:
        trainer_errors.append(f"TrainingArguments fallback failed: {exc}")

    joined = " | ".join(trainer_errors) if trainer_errors else "unknown error"
    raise WorkerContractError(f"failed to initialize SFTTrainer: {joined}")


def _save_merged_checkpoint(model: Any, tokenizer: Any, output_dir: Path) -> Path:
    merged_dir = output_dir / "merged_model"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged = _unwrap_model(model)
    if hasattr(merged, "merge_and_unload"):
        merged = merged.merge_and_unload()
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    return merged_dir


def _unwrap_model(model: Any) -> Any:
    unwrapped = model
    if hasattr(unwrapped, "module"):
        return getattr(unwrapped, "module")
    return unwrapped


_VISION_PARAM_HINTS = (
    "visual",
    "vision",
    "image_processor",
    "image_encoder",
    "patch_embed",
    "vit",
)
_QWEN35_NESTED_LANGUAGE_PREFIX = "model.language_model.language_model.language_model."
_QWEN35_BASE_LANGUAGE_PREFIX = "model.language_model."


def _prepare_full_weight_model(model: Any) -> Any:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    # Unfreeze only the language-side parameters. Qwen3.5 VL has a vision
    # encoder that never participates in text-only forward, so leaving its
    # params trainable causes DDP to abort with "Expected to have finished
    # reduction in the prior iteration" because no gradient ever lands on
    # those buckets.
    for name, param in model.named_parameters():
        lower = name.lower()
        if any(hint in lower for hint in _VISION_PARAM_HINTS):
            param.requires_grad = False
        else:
            param.requires_grad = True
    return model


def _normalize_full_weight_checkpoint_keys(checkpoint_dir: Path) -> bool:
    """Fix Unsloth wrapper key nesting after full-weight Qwen3.5 saves.

    FastVisionModel/FastLanguageModel can leave a text model wrapped under
    repeated ``language_model`` modules. Saving that wrapper emits keys like
    ``model.language_model.language_model.language_model.layers...`` while the
    base checkpoint and downstream vLLM loader expect
    ``model.language_model.layers...``. Rewrite only this exact all-nested
    pattern so other model layouts stay untouched.
    """

    safetensor_paths = sorted(checkpoint_dir.glob("*.safetensors"))
    if not safetensor_paths:
        return False

    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ModuleNotFoundError as exc:
        raise WorkerContractError("safetensors package is required to normalize checkpoint keys") from exc

    saw_nested = False
    for path in safetensor_paths:
        with safe_open(path, framework="pt") as handle:
            for key in handle.keys():
                if key.startswith(_QWEN35_NESTED_LANGUAGE_PREFIX):
                    saw_nested = True

    if not saw_nested:
        return False

    for path in safetensor_paths:
        tensors: dict[str, Any] = {}
        metadata: dict[str, str] | None = None
        with safe_open(path, framework="pt") as handle:
            raw_metadata = handle.metadata()
            if raw_metadata:
                metadata = {str(key): str(value) for key, value in raw_metadata.items()}
            for key in handle.keys():
                normalized_key = key
                if key.startswith(_QWEN35_NESTED_LANGUAGE_PREFIX):
                    normalized_key = _QWEN35_BASE_LANGUAGE_PREFIX + key[len(_QWEN35_NESTED_LANGUAGE_PREFIX) :]
                if normalized_key in tensors:
                    raise WorkerContractError(
                        f"duplicate checkpoint key after normalization: {normalized_key}"
                    )
                tensors[normalized_key] = handle.get_tensor(key)
        tmp_path = path.with_name(f".{path.name}.tmp")
        try:
            save_file(tensors, str(tmp_path), metadata=metadata)
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    for index_path in checkpoint_dir.glob("*.safetensors.index.json"):
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if isinstance(weight_map, dict):
            normalized_map: dict[str, Any] = {}
            for key, value in weight_map.items():
                normalized_key = str(key)
                if normalized_key.startswith(_QWEN35_NESTED_LANGUAGE_PREFIX):
                    normalized_key = (
                        _QWEN35_BASE_LANGUAGE_PREFIX
                        + normalized_key[len(_QWEN35_NESTED_LANGUAGE_PREFIX) :]
                    )
                if normalized_key in normalized_map:
                    raise WorkerContractError(
                        f"duplicate index key after normalization: {normalized_key}"
                    )
                normalized_map[normalized_key] = value
            payload["weight_map"] = normalized_map
            index_path.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )

    return True


def _assert_full_weight_checkpoint_keys(checkpoint_dir: Path) -> None:
    safetensor_paths = sorted(checkpoint_dir.glob("*.safetensors"))
    if not safetensor_paths:
        raise WorkerContractError(f"full-weight checkpoint has no safetensors: {checkpoint_dir}")
    try:
        from safetensors import safe_open
    except ModuleNotFoundError as exc:
        raise WorkerContractError("safetensors package is required to validate checkpoint keys") from exc

    saw_base_style = False
    saw_nested = False
    for path in safetensor_paths:
        with safe_open(path, framework="pt") as handle:
            for key in handle.keys():
                if key.startswith(_QWEN35_BASE_LANGUAGE_PREFIX):
                    saw_base_style = True
                if key.startswith(_QWEN35_NESTED_LANGUAGE_PREFIX):
                    saw_nested = True
    if saw_nested:
        raise WorkerContractError(
            "full-weight checkpoint still contains nested language_model keys after save"
        )
    if not saw_base_style:
        raise WorkerContractError(
            "full-weight checkpoint missing base-style model.language_model.* keys"
        )


def _run_sft_train(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    runtime: _TrainRuntime,
    lora_cfg: Mapping[str, Any],
    train_cfg: Mapping[str, Any],
    seed: int,
    mode: str = "lora",
    save_merged_checkpoint: bool = True,
    logging_cfg: Mapping[str, Any] | None = None,
    phase: str | None = None,
    is_main_process: bool = True,
    prompts_cfg: Mapping[str, Any] | None = None,
) -> tuple[Path, Path | None]:
    train_cfg_with_prompts = dict(train_cfg)
    if prompts_cfg:
        train_cfg_with_prompts["runtime_prompts"] = dict(prompts_cfg)
    response_template = _resolve_response_template(
        train_cfg_with_prompts, phase=phase or "training"
    )
    model, tokenizer = _load_unsloth_model(
        runtime, full_finetuning=(mode == "full_weight")
    )
    if mode == "full_weight":
        model = _prepare_full_weight_model(model)
    else:
        model = _ensure_lora_model(model=model, runtime=runtime, lora_cfg=lora_cfg, seed=seed)
    dataset = _build_dataset(
        rows,
        tokenizer,
        prompts_cfg=prompts_cfg,
        max_seq_length=runtime.max_seq_length,
    )
    trainer = _instantiate_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=output_dir,
        train_cfg=train_cfg,
        max_seq_length=runtime.max_seq_length,
        response_template=response_template,
        logging_cfg={
            **_as_dict(logging_cfg or {}),
            "phase": phase,
        },
        is_main_process=is_main_process,
    )
    trainer.train()

    if mode == "full_weight":
        checkpoint_dir = output_dir / "full_weight_model"
        if is_main_process:
            save_model = _unwrap_model(model)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_model.save_pretrained(str(checkpoint_dir))
            _normalize_full_weight_checkpoint_keys(checkpoint_dir)
            _assert_full_weight_checkpoint_keys(checkpoint_dir)
            tokenizer.save_pretrained(str(checkpoint_dir))
        return checkpoint_dir, None
    else:
        adapter_dir = output_dir / "main_adapter"
        if is_main_process:
            save_model = _unwrap_model(model)
            adapter_dir.mkdir(parents=True, exist_ok=True)
            save_model.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))

        merged_dir: Path | None = None
        if save_merged_checkpoint and is_main_process:
            merged_dir = _save_merged_checkpoint(model, tokenizer, output_dir)
        return adapter_dir, merged_dir


def _phase(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "update-base"
    return str(rows[0].get("phase", "update-base"))


def _resolve_runtime_prompts_cfg(row: Mapping[str, Any]) -> dict[str, Any]:
    runtime_cfg = _as_dict(row.get("runtime_config"))
    prompts_cfg = _as_dict(runtime_cfg.get("prompts"))
    if prompts_cfg:
        return prompts_cfg
    return _as_dict(row.get("prompts"))


def _collapse_train(
    rows: list[dict[str, Any]],
    *,
    is_main_process: bool,
) -> list[dict[str, Any]]:
    first = rows[0]
    runtime = _resolve_train_runtime(first)
    seed = int(first.get("subset_idx", 0) or 0)
    adapter_path = Path(str(first.get("adapter_path", "collapse_adapter")))
    adapter_path.mkdir(parents=True, exist_ok=True)

    collapse_cfg = _as_dict(first.get("training_config"))
    prompts_cfg = _resolve_runtime_prompts_cfg(first)
    # Collapse config does not include target modules by default; use project baseline.
    lora_cfg = {
        "rank": collapse_cfg.get("rank", 4),
        "alpha": collapse_cfg.get("alpha", collapse_cfg.get("rank", 4) * 2),
        "dropout": collapse_cfg.get("dropout", 0.0),
        "bias": collapse_cfg.get("bias", "none"),
        "target_modules": collapse_cfg.get("target_modules", list(_DEFAULT_LORA_TARGETS)),
    }
    batching_cfg = _as_dict(collapse_cfg.get("batching"))
    train_cfg = {
        "num_train_epochs": collapse_cfg.get("num_train_epochs", 1),
        "learning_rate": collapse_cfg.get("learning_rate", 5e-3),
        "per_device_train_batch_size": _as_positive_int(batching_cfg.get("per_device_train_batch_size"), 1),
        "gradient_accumulation_steps": _as_positive_int(batching_cfg.get("gradient_accumulation_steps"), 1),
        "optimizer": {
            "learning_rate": collapse_cfg.get("learning_rate", 5e-3),
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
            "lr_scheduler_type": "cosine",
            "optim": "adamw_torch",
            "max_grad_norm": 1.0,
        },
        "batching": {
            "per_device_train_batch_size": _as_positive_int(batching_cfg.get("per_device_train_batch_size"), 1),
            "gradient_accumulation_steps": _as_positive_int(batching_cfg.get("gradient_accumulation_steps"), 1),
            "packing": bool(batching_cfg.get("packing", False)),
        },
    }

    # Save collapse adapter as output_dir/main_adapter first, then move files up.
    scratch_dir = adapter_path / "_train_tmp"
    try:
        adapter_tmp, _ = _run_sft_train(
            rows=rows,
            output_dir=scratch_dir,
            runtime=runtime,
            lora_cfg=lora_cfg,
            train_cfg=train_cfg,
            seed=seed,
            save_merged_checkpoint=False,
            logging_cfg=_as_dict(first.get("logging_config")),
            phase="train-collapse-lora",
            is_main_process=is_main_process,
            prompts_cfg=prompts_cfg,
        )
    except WorkerContractError as exc:
        if not _is_context_overflow_skip_error(exc):
            raise
        reason = str(exc)
        if is_main_process:
            _write_noop_collapse_marker(adapter_path, reason=reason)
        return [
            {
                "status": "ok",
                "adapter_path": str(adapter_path),
                "trained_rows": 0,
                "backend": "unsloth",
                "error": None,
                "no_op": True,
                "reason": reason,
            }
        ]
    if is_main_process and adapter_tmp.exists():
        for item in adapter_tmp.iterdir():
            target = adapter_path / item.name
            if target.exists():
                if target.is_file():
                    target.unlink()
                else:
                    continue
            item.replace(target)

    return [
        {
            "status": "ok",
            "adapter_path": str(adapter_path),
            "trained_rows": len(rows),
            "backend": "unsloth",
            "error": None,
        }
    ]


def _unload(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adapter_path = str(rows[0].get("adapter_path", "") if rows else "")
    if not adapter_path:
        raise WorkerContractError("unload-collapse-lora requires adapter_path")
    path = Path(adapter_path)
    if not path.exists():
        raise WorkerContractError(f"collapse adapter path not found: {adapter_path}")
    if not (path / "adapter_config.json").exists() and not _is_noop_collapse_adapter_path(path):
        raise WorkerContractError(
            f"collapse adapter path missing adapter_config.json: {adapter_path}"
        )
    registry_hash = hashlib.sha256(adapter_path.encode("utf-8")).hexdigest()
    return [
        {
            "status": "ok",
            "adapter_path": adapter_path,
            "clean_base": True,
            "active_adapters": [],
            "collapse_merged": False,
            "adapter_registry_hash": registry_hash,
            "verified_adapter_path": adapter_path,
            "backend": "unsloth",
            "error": None,
        }
    ]


def _update_base(
    rows: list[dict[str, Any]],
    *,
    is_main_process: bool,
) -> list[dict[str, Any]]:
    if not rows:
        raise WorkerContractError("update-base received empty training rows")
    first = rows[0]
    runtime = _resolve_train_runtime(first)
    seed = int(first.get("subset_idx", 0) or 0)
    output_dir = Path(str(first.get("output_dir", "train_final")))
    output_dir.mkdir(parents=True, exist_ok=True)

    training_cfg = _as_dict(first.get("training_config"))
    prompts_cfg = _resolve_runtime_prompts_cfg(first)
    update_mode = str(training_cfg.get("mode", "lora"))
    base_lora_cfg = _as_dict(training_cfg.get("lora"))
    lora_cfg = {
        "rank": base_lora_cfg.get("rank", 16),
        "alpha": base_lora_cfg.get("alpha", 32),
        "dropout": base_lora_cfg.get("dropout", 0.0),
        "bias": base_lora_cfg.get("bias", "none"),
        "target_modules": base_lora_cfg.get("target_modules", list(_DEFAULT_LORA_TARGETS)),
    }
    try:
        adapter_dir, merged_dir = _run_sft_train(
            rows=rows,
            output_dir=output_dir,
            runtime=runtime,
            lora_cfg=lora_cfg,
            train_cfg=training_cfg,
            seed=seed,
            mode=update_mode,
            save_merged_checkpoint=(update_mode != "full_weight"),
            logging_cfg=_as_dict(first.get("logging_config")),
            phase="update-base",
            is_main_process=is_main_process,
            prompts_cfg=prompts_cfg,
        )
    except WorkerContractError as exc:
        if not _is_context_overflow_skip_error(exc):
            raise
        reason = str(exc)
        fallback_checkpoint = first.get("base_checkpoint")
        if isinstance(fallback_checkpoint, str) and fallback_checkpoint.strip():
            checkpoint_path = fallback_checkpoint.strip()
        else:
            checkpoint_path = str(output_dir)
        return [
            {
                "status": "ok",
                "checkpoint_path": checkpoint_path,
                "trained_rows": 0,
                "backend": "unsloth",
                "error": None,
                "no_op": True,
                "reason": reason,
            }
        ]
    effective_checkpoint = merged_dir if merged_dir is not None else adapter_dir
    checkpoint_state: dict[str, Any] = {
        "status": "ok",
        "mode": update_mode,
        "trained_rows": len(rows),
        "backend": "unsloth",
    }
    if update_mode == "full_weight":
        checkpoint_state["full_weight_path"] = str(adapter_dir)
    else:
        checkpoint_state["adapter_path"] = str(adapter_dir)
        checkpoint_state["merged_checkpoint_path"] = str(merged_dir) if merged_dir is not None else None
    if is_main_process:
        (output_dir / "worker_checkpoint_state.json").write_text(
            json.dumps(checkpoint_state, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    return [
        {
            "status": "ok",
            "checkpoint_path": str(effective_checkpoint),
            "trained_rows": len(rows),
            "backend": "unsloth",
            "error": None,
        }
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_worker_args(description="Real training worker", argv=argv)
    ddp = _init_ddp_runtime()

    try:
        requests = [dict(row) for row in read_jsonl(args.input_path)]
        phase = str(args.phase or _phase(requests))
        schema = validate_phase_request_rows(requests, args=args, context="training")

        if phase == "train-collapse-lora":
            responses = _collapse_train(requests, is_main_process=ddp.is_main_process)
        elif phase == "unload-collapse-lora":
            responses = _unload(requests)
        elif phase == "update-base":
            responses = _update_base(requests, is_main_process=ddp.is_main_process)
        else:
            raise WorkerContractError(f"unsupported training phase: {phase}")

        validate_phase_response_rows(responses, schema=schema, context="training")
        if ddp.is_main_process:
            write_jsonl(args.output_path, responses, ensure_ascii=False)
        _ddp_barrier(ddp)
        return 0
    finally:
        _shutdown_ddp_runtime(ddp)


if __name__ == "__main__":
    raise SystemExit(main())
