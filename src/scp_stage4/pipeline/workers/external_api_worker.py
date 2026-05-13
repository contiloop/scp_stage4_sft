"""Real external API worker for subprocess runtime.

Currently supports OpenAI provider through the Responses API.
"""

from __future__ import annotations

import os
import time
from typing import Any, Mapping

from scp_stage4.data import read_jsonl, write_jsonl
from scp_stage4.pipeline.prompting import (
    PromptConfigError,
    render_teacher_user_prompt,
    teacher_system_prompt,
)
from scp_stage4.pipeline.workers.common import (
    WorkerContractError,
    parse_worker_args,
    validate_phase_request_rows,
    validate_phase_response_rows,
)

_LABELS = {"no_change", "minor_edit", "major_edit", "rewrite", "invalid"}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _split_label_and_text(output_text: str) -> tuple[str, str]:
    lines = [line.strip() for line in output_text.splitlines() if line.strip()]
    if not lines:
        return "invalid", "empty API response"

    label = lines[0].lower()
    if label not in _LABELS:
        # Be tolerant to prefixes like "label: minor_edit".
        for candidate in _LABELS:
            if candidate in label:
                label = candidate
                break
        else:
            label = "minor_edit"

    rest = "\n".join(lines[1:]).strip()
    if not rest:
        rest = "출력이 비어 있어 수정이 필요합니다."
    return label, rest


def _extract_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    usage_dict = dict(usage) if isinstance(usage, Mapping) else {}

    input_tokens = usage_dict.get("input_tokens", getattr(usage, "input_tokens", 0))
    output_tokens = usage_dict.get("output_tokens", getattr(usage, "output_tokens", 0))
    total_tokens = usage_dict.get("total_tokens", getattr(usage, "total_tokens", 0))
    if not total_tokens:
        total_tokens = int(input_tokens or 0) + int(output_tokens or 0)

    return {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }


def _resolve_api_key_env(row: Mapping[str, Any]) -> str:
    runtime_cfg = _as_dict(row.get("runtime_config"))
    external_api_cfg = _as_dict(runtime_cfg.get("external_api"))
    primary_cfg = _as_dict(external_api_cfg.get("primary"))
    env_name = str(primary_cfg.get("api_key_env", "")).strip()
    if env_name:
        return env_name

    provider = str(row.get("provider", "openai")).lower()
    defaults = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "qwen": "QWEN_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    return defaults.get(provider, "OPENAI_API_KEY")


def _openai_call(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise WorkerContractError("openai package is required for external_api worker") from exc

    env_name = _resolve_api_key_env(row)
    api_key = os.environ.get(env_name)
    if not api_key:
        raise WorkerContractError(f"missing API key env var: {env_name}")

    model = str(row.get("model", "")).strip()
    if not model:
        raise WorkerContractError("external_api request row missing model")

    runtime_cfg = _as_dict(row.get("runtime_config"))
    prompts_cfg = _as_dict(runtime_cfg.get("prompts"))
    try:
        prompt = render_teacher_user_prompt(prompts=prompts_cfg, row=row)
        system_prompt = teacher_system_prompt(prompts_cfg)
    except PromptConfigError as exc:
        raise WorkerContractError(str(exc)) from exc
    client = OpenAI(api_key=api_key)
    started = time.perf_counter()
    response = client.responses.create(
        model=model,
        temperature=0.0,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
        ],
    )
    latency_ms = (time.perf_counter() - started) * 1000.0

    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str) or not output_text.strip():
        raise WorkerContractError("OpenAI response did not contain output_text")

    teacher_label, payload_text = _split_label_and_text(output_text)
    status = "ok" if teacher_label != "invalid" else "filtered"

    return {
        "request_id": str(row.get("request_id", "")),
        "status": status,
        "gold": payload_text if status == "ok" else None,
        "teacher_label": teacher_label,
        "usage": _extract_usage(response),
        "cost": {"currency": "USD", "estimated": 0.0},
        "latency_ms": round(latency_ms, 3),
        "attempt": 1,
        "reason": payload_text if status != "ok" else None,
        "error": None,
    }


def _fallback_error_response(row: Mapping[str, Any], message: str) -> dict[str, Any]:
    return {
        "request_id": str(row.get("request_id", "")),
        "status": "failed",
        "gold": None,
        "teacher_label": "invalid",
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "cost": {"currency": "USD", "estimated": 0.0},
        "latency_ms": 0.0,
        "attempt": 1,
        "reason": message,
        "error": message,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_worker_args(description="Real external API worker", argv=argv)

    requests = [dict(row) for row in read_jsonl(args.input_path)]
    schema = validate_phase_request_rows(requests, args=args, context="external_api")
    responses: list[dict[str, Any]] = []
    for row in requests:
        provider = str(row.get("provider", "openai")).lower()
        try:
            if provider != "openai":
                raise WorkerContractError(
                    f"provider={provider!r} is not implemented yet in external_api worker"
                )
            responses.append(_openai_call(row))
        except Exception as exc:
            responses.append(_fallback_error_response(row, str(exc)))

    validate_phase_response_rows(responses, schema=schema, context="external_api")
    write_jsonl(args.output_path, responses, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
