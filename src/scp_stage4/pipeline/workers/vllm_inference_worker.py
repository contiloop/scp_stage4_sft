"""vLLM offline batch inference worker for infer-q1 / infer-q2.

Drop-in replacement for inference_worker.py using vLLM's continuous
batching engine.  Activated by setting inference.runtime.subprocess.command
to ["python3", "-m", "scp_stage4.pipeline.workers.vllm_inference_worker"].
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

from scp_stage4.data import read_jsonl, write_jsonl
from scp_stage4.pipeline.workers.common import (
    WorkerContractError,
    parse_worker_args,
    validate_phase_request_rows,
    validate_phase_response_rows,
)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _build_prompt(source: str) -> str:
    return (
        "You are a professional English to Korean translator.\n"
        "Translate the English input into natural Korean.\n"
        "Return only the Korean translation.\n\n"
        f"English:\n{source}\n\nKorean:\n"
    )


def _is_lora_adapter_path(path: Path) -> bool:
    return path.exists() and (path / "adapter_config.json").exists()


def _resolve_model_name(request: Mapping[str, Any]) -> str:
    runtime_cfg = _as_dict(request.get("runtime_config"))
    model_cfg = _as_dict(runtime_cfg.get("model"))
    name = str(model_cfg.get("name", "")).strip()
    if not name:
        raise WorkerContractError("runtime_config.model.name is required")
    return name


def _resolve_vllm_kwargs(request: Mapping[str, Any]) -> dict[str, Any]:
    runtime_cfg = _as_dict(request.get("runtime_config"))
    model_cfg = _as_dict(runtime_cfg.get("model"))

    max_seq_length = model_cfg.get("max_seq_length") or model_cfg.get("max_length")
    if isinstance(max_seq_length, bool) or not isinstance(max_seq_length, int):
        max_seq_length = 8192

    dtype_raw = str(model_cfg.get("dtype", "auto")).strip().lower()
    dtype_map = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float32": "float32",
    }
    dtype = dtype_map.get(dtype_raw, "auto")

    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))

    kwargs: dict[str, Any] = {
        "max_model_len": max_seq_length,
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }

    gpu_memory_utilization = model_cfg.get("gpu_memory_utilization")
    if isinstance(gpu_memory_utilization, (int, float)) and 0 < gpu_memory_utilization <= 1:
        kwargs["gpu_memory_utilization"] = float(gpu_memory_utilization)

    tensor_parallel_size = model_cfg.get("tensor_parallel_size")
    if isinstance(tensor_parallel_size, int) and tensor_parallel_size > 1:
        kwargs["tensor_parallel_size"] = tensor_parallel_size

    return kwargs


def _resolve_lora_path(request: Mapping[str, Any]) -> Path | None:
    """Return the LoRA adapter path for Q2, or None for Q1."""
    q_tag = str(request.get("q_tag", "q1"))
    if q_tag != "q2":
        return None
    collapse_adapter = request.get("collapse_adapter")
    if not isinstance(collapse_adapter, str) or not collapse_adapter.strip():
        raise WorkerContractError("infer-q2 requires non-empty collapse_adapter path")
    p = Path(collapse_adapter)
    if not _is_lora_adapter_path(p):
        raise WorkerContractError(
            "collapse adapter is missing adapter_config.json; "
            "run train-collapse-lora first"
        )
    return p


def _resolve_base_lora_path(request: Mapping[str, Any]) -> Path | None:
    base_checkpoint = request.get("base_checkpoint")
    if not isinstance(base_checkpoint, str) or not base_checkpoint.strip():
        return None
    p = Path(base_checkpoint)
    if _is_lora_adapter_path(p):
        return p
    return None


def _build_sampling_params(request: Mapping[str, Any]) -> Any:
    from vllm import SamplingParams

    decoding = _as_dict(request.get("decoding"))
    max_tokens = int(decoding.get("max_new_tokens", 256) or 256)
    if max_tokens <= 0:
        max_tokens = 256

    do_sample = bool(decoding.get("do_sample", False))
    temperature = float(decoding.get("temperature", 0.0) or 0.0)
    top_p_raw = decoding.get("top_p")
    top_p = float(top_p_raw) if isinstance(top_p_raw, (int, float)) else 1.0

    if not do_sample or temperature == 0.0:
        return SamplingParams(max_tokens=max_tokens, temperature=0.0)

    return SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )


def _load_engine(
    requests: list[Mapping[str, Any]],
) -> tuple[Any, Any | None, Any | None]:
    """Load vLLM engine and optional LoRA requests.

    Returns (llm, base_lora_request, collapse_lora_request).
    """
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    first = requests[0]
    model_name = _resolve_model_name(first)
    vllm_kwargs = _resolve_vllm_kwargs(first)

    base_lora = _resolve_base_lora_path(first)
    collapse_lora = _resolve_lora_path(first)
    need_lora = base_lora is not None or collapse_lora is not None

    if need_lora:
        vllm_kwargs["enable_lora"] = True
        max_lora_rank = _as_dict(
            _as_dict(first.get("runtime_config")).get("model")
        ).get("max_lora_rank", 64)
        if isinstance(max_lora_rank, int) and max_lora_rank > 0:
            vllm_kwargs["max_lora_rank"] = max_lora_rank

    print(
        f"[vllm-inference-worker] loading model={model_name} "
        f"lora={need_lora} kwargs={vllm_kwargs}",
        file=sys.stderr,
    )

    # Text-only task: disable multimodal to avoid image processor errors
    # on VLM architectures (e.g. Qwen3.5).
    vllm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}

    try:
        llm = LLM(model=model_name, **vllm_kwargs)
    except Exception:
        vllm_kwargs.pop("limit_mm_per_prompt", None)
        vllm_kwargs["disable_mm_preprocessor_cache"] = True
        llm = LLM(model=model_name, **vllm_kwargs)

    base_lora_request = None
    if base_lora is not None:
        base_lora_request = LoRARequest(
            lora_name="base_update",
            lora_int_id=1,
            lora_path=str(base_lora),
        )
        print(
            f"[vllm-inference-worker] base LoRA adapter: {base_lora}",
            file=sys.stderr,
        )

    collapse_lora_request = None
    if collapse_lora is not None:
        collapse_lora_request = LoRARequest(
            lora_name="collapse_probe",
            lora_int_id=2,
            lora_path=str(collapse_lora),
        )
        print(
            f"[vllm-inference-worker] collapse LoRA adapter: {collapse_lora}",
            file=sys.stderr,
        )

    return llm, base_lora_request, collapse_lora_request


def _generate_all(
    *,
    llm: Any,
    requests: list[Mapping[str, Any]],
    base_lora_request: Any | None,
    collapse_lora_request: Any | None,
) -> list[dict[str, Any]]:
    prompts: list[str] = []
    sampling_params_list: list[Any] = []
    for row in requests:
        source = str(row.get("source", "")).strip()
        if not source:
            raise WorkerContractError(
                f"inference request row missing source text (id={row.get('id')})"
            )
        prompts.append(_build_prompt(source))
        sampling_params_list.append(_build_sampling_params(row))

    q_tag = str(requests[0].get("q_tag", "q1"))
    lora_request = None
    if q_tag == "q2" and collapse_lora_request is not None:
        lora_request = collapse_lora_request
    elif base_lora_request is not None:
        lora_request = base_lora_request

    first_sp = sampling_params_list[0]
    all_same_params = all(
        type(sp) is type(first_sp)
        and getattr(sp, "temperature", None) == getattr(first_sp, "temperature", None)
        and getattr(sp, "top_p", None) == getattr(first_sp, "top_p", None)
        and getattr(sp, "max_tokens", None) == getattr(first_sp, "max_tokens", None)
        for sp in sampling_params_list[1:]
    )

    print(
        f"[vllm-inference-worker] generating {len(prompts)} prompts "
        f"(q_tag={q_tag}, lora={'yes' if lora_request else 'no'}, "
        f"uniform_params={all_same_params})",
        file=sys.stderr,
    )

    generate_kwargs: dict[str, Any] = {}
    if lora_request is not None:
        generate_kwargs["lora_request"] = lora_request

    if all_same_params:
        outputs = llm.generate(
            prompts,
            sampling_params=sampling_params_list[0],
            **generate_kwargs,
        )
    else:
        outputs = llm.generate(
            prompts,
            sampling_params=sampling_params_list,
            **generate_kwargs,
        )

    responses: list[dict[str, Any]] = []
    for idx, output in enumerate(outputs):
        req = requests[idx]
        req_id = str(req.get("id", ""))
        text = output.outputs[0].text.strip() if output.outputs else ""
        if not text:
            responses.append({
                "id": req_id,
                "order_idx": int(req.get("order_idx", idx)),
                "status": "failed",
                "mt": "",
                "error": "vllm generation returned empty translation",
            })
            continue
        responses.append({
            "id": req_id,
            "order_idx": int(req.get("order_idx", idx)),
            "status": "ok",
            "mt": text,
            "error": None,
        })

    ok_count = sum(1 for r in responses if r["status"] == "ok")
    print(
        f"[vllm-inference-worker] done: {ok_count}/{len(responses)} ok",
        file=sys.stderr,
    )
    return responses


def main(argv: list[str] | None = None) -> int:
    args = parse_worker_args(description="vLLM inference worker", argv=argv)

    requests = read_jsonl(args.input_path)
    schema = validate_phase_request_rows(requests, args=args, context="inference")
    if not requests:
        write_jsonl(args.output_path, [], ensure_ascii=False)
        return 0

    try:
        llm, base_lora_request, collapse_lora_request = _load_engine(requests)
        responses = _generate_all(
            llm=llm,
            requests=requests,
            base_lora_request=base_lora_request,
            collapse_lora_request=collapse_lora_request,
        )
    except Exception as exc:
        print(f"[vllm-inference-worker] fatal: {exc}", file=sys.stderr)
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
