"""Real inference worker for subprocess runtime.

This worker executes local model generation for infer-q1 / infer-q2 using
Transformers and optional PEFT adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from scp_stage4.data import read_jsonl, write_jsonl
from scp_stage4.pipeline.workers.common import (
    WorkerContractError,
    parse_worker_args,
    validate_phase_request_rows,
    validate_phase_response_rows,
)


@dataclass(frozen=True)
class _ModelRuntime:
    model_ref: str
    tokenizer_ref: str
    trust_remote_code: bool
    torch_dtype: torch.dtype | str | None
    max_seq_length: int | None
    padding_side: str


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
            # Let Transformers decide for uncommon local checkpoint layouts.
            model_ref = str(cp)

    tokenizer_ref = model_name
    if isinstance(model_ref, str) and Path(model_ref).exists() and _is_model_checkpoint_path(Path(model_ref)):
        tokenizer_ref = model_ref

    max_seq_length = model_cfg.get("max_seq_length")
    if isinstance(max_seq_length, bool) or not isinstance(max_seq_length, int):
        max_seq_length = None

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


def _build_prompt(source: str) -> str:
    return (
        "You are a professional English to Korean translator.\n"
        "Translate the English input into natural Korean.\n"
        "Return only the Korean translation.\n\n"
        f"English:\n{source}\n\nKorean:\n"
    )


def _load_model(request: Mapping[str, Any]) -> tuple[Any, Any]:
    runtime = _resolve_runtime(request)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        runtime.tokenizer_ref,
        trust_remote_code=runtime.trust_remote_code,
    )
    tokenizer.padding_side = runtime.padding_side
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": runtime.trust_remote_code,
        "device_map": "auto",
    }
    if runtime.torch_dtype is not None:
        model_kwargs["torch_dtype"] = runtime.torch_dtype

    model = AutoModelForCausalLM.from_pretrained(runtime.model_ref, **model_kwargs)

    base_checkpoint = request.get("base_checkpoint")
    if isinstance(base_checkpoint, str) and base_checkpoint.strip():
        cp = Path(base_checkpoint)
        if _is_lora_adapter_path(cp):
            try:
                from peft import PeftModel
            except ModuleNotFoundError as exc:
                raise WorkerContractError(
                    "peft is required to load base update adapters for inference"
                ) from exc
            model = PeftModel.from_pretrained(
                model,
                str(cp),
                adapter_name="base_update",
                is_trainable=False,
            )
            model.set_adapter("base_update")

    q_tag = str(request.get("q_tag", "q1"))
    collapse_adapter = request.get("collapse_adapter")
    if q_tag == "q2":
        if not isinstance(collapse_adapter, str) or not collapse_adapter.strip():
            raise WorkerContractError("infer-q2 requires non-empty collapse_adapter path")
        cp = Path(collapse_adapter)
        if not _is_lora_adapter_path(cp):
            raise WorkerContractError(
                f"collapse adapter path is missing adapter_config.json: {cp}"
            )
        try:
            from peft import PeftModel
        except ModuleNotFoundError as exc:
            raise WorkerContractError(
                "peft is required to load collapse adapter for infer-q2"
            ) from exc
        if hasattr(model, "load_adapter"):
            model.load_adapter(str(cp), adapter_name="collapse", is_trainable=False)
            model.set_adapter("collapse")
        else:
            model = PeftModel.from_pretrained(
                model,
                str(cp),
                adapter_name="collapse",
                is_trainable=False,
            )
            model.set_adapter("collapse")

    model.eval()
    return model, tokenizer


def _generate_one(
    *,
    model: Any,
    tokenizer: Any,
    request: Mapping[str, Any],
) -> str:
    source = str(request.get("source", "")).strip()
    if not source:
        raise WorkerContractError("inference request row missing source text")

    prompt = _build_prompt(source)
    decoding_cfg = _as_dict(request.get("decoding"))
    max_new_tokens = int(decoding_cfg.get("max_new_tokens", 256) or 256)
    if max_new_tokens <= 0:
        max_new_tokens = 256

    do_sample = bool(decoding_cfg.get("do_sample", False))
    temperature = float(decoding_cfg.get("temperature", 0.0) or 0.0)
    top_p_raw = decoding_cfg.get("top_p", None)
    top_p = float(top_p_raw) if isinstance(top_p_raw, (int, float)) else None

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = int(inputs["input_ids"].shape[1])

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample and temperature > 0:
        generate_kwargs["temperature"] = temperature
        if top_p is not None:
            generate_kwargs["top_p"] = top_p
    else:
        generate_kwargs["temperature"] = 0.0

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)

    text = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    if not text:
        raise WorkerContractError("inference generation returned empty translation")
    return text


def main(argv: list[str] | None = None) -> int:
    args = parse_worker_args(description="Real inference worker", argv=argv)

    requests = read_jsonl(args.input_path)
    schema = validate_phase_request_rows(requests, args=args, context="inference")
    if not requests:
        write_jsonl(args.output_path, [], ensure_ascii=False)
        return 0

    model, tokenizer = _load_model(requests[0])
    responses = []
    for row in requests:
        req_id = str(row.get("id", ""))
        try:
            mt = _generate_one(model=model, tokenizer=tokenizer, request=row)
            responses.append({"id": req_id, "status": "ok", "mt": mt, "error": None})
        except Exception as exc:
            responses.append(
                {
                    "id": req_id,
                    "status": "failed",
                    "mt": "",
                    "error": str(exc),
                }
            )

    validate_phase_response_rows(responses, schema=schema, context="inference")
    write_jsonl(args.output_path, responses, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
