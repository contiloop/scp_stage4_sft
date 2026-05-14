"""Real external API worker for subprocess runtime.

Supports three providers:
- ``openai``  (Responses API; GPT-5.x reasoning models with ``reasoning.effort``)
- ``anthropic`` (Messages API; Claude 4.x adaptive thinking)
- ``gemini`` (REST; Gemini 3.x with optional ``thinkingConfig``)

Provider selection per row is encoded in the input JSONL by upstream pipeline
code (see ``scp_stage4.pipeline.routing``). Each row carries ``provider``,
``model`` and an optional ``model_params`` dict; the worker dispatches based on
those fields.

The worker also:
- runs requests in parallel via :class:`concurrent.futures.ThreadPoolExecutor`,
- enforces per-provider concurrency caps via semaphores,
- prints a tqdm-style progress bar to stderr,
- retries transient failures (5xx / 429 / 503) with exponential backoff,
- best-effort recovers from 400 responses by stripping the field implicated
  in the error message (e.g. ``top_p``, ``thinkingConfig``).
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

_DEFAULT_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

_PROVIDER_CONCURRENCY_DEFAULT = {
    "openai": 4,
    "anthropic": 4,
    "gemini": 6,
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _split_label_and_text(output_text: str) -> tuple[str, str]:
    lines = [line.strip() for line in output_text.splitlines() if line.strip()]
    if not lines:
        return "invalid", "empty API response"

    label = lines[0].lower()
    if label not in _LABELS:
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


def _resolve_api_key_env(row: Mapping[str, Any]) -> str:
    runtime_cfg = _as_dict(row.get("runtime_config"))
    external_api_cfg = _as_dict(runtime_cfg.get("external_api"))
    providers_cfg = _as_dict(external_api_cfg.get("providers"))
    provider = str(row.get("provider", "")).strip().lower()
    if provider:
        provider_cfg = _as_dict(providers_cfg.get(provider))
        env_name = str(provider_cfg.get("api_key_env", "")).strip()
        if env_name:
            return env_name
    # legacy path
    primary_cfg = _as_dict(external_api_cfg.get("primary"))
    env_name = str(primary_cfg.get("api_key_env", "")).strip()
    if env_name:
        return env_name
    return _DEFAULT_API_KEY_ENV.get(provider, "OPENAI_API_KEY")


def _provider_timeout(row: Mapping[str, Any], provider: str) -> float:
    runtime_cfg = _as_dict(row.get("runtime_config"))
    timeouts = _as_dict(_as_dict(runtime_cfg.get("external_api")).get("timeouts"))
    key_map = {
        "openai": "openai_sec",
        "anthropic": "anthropic_sec",
        "gemini": "gemini_sec",
    }
    default = {"openai": 120, "anthropic": 180, "gemini": 120}.get(provider, 120)
    try:
        return float(timeouts.get(key_map.get(provider, ""), default))
    except (TypeError, ValueError):
        return float(default)


def _prompt_pair(row: Mapping[str, Any]) -> tuple[str, str]:
    runtime_cfg = _as_dict(row.get("runtime_config"))
    prompts_cfg = _as_dict(runtime_cfg.get("prompts"))
    try:
        user_prompt = render_teacher_user_prompt(prompts=prompts_cfg, row=row)
        system_prompt = teacher_system_prompt(prompts_cfg)
    except PromptConfigError as exc:
        raise WorkerContractError(str(exc)) from exc
    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------------


def _openai_call(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise WorkerContractError("openai package is required for external_api worker") from exc

    api_key = os.environ.get(_resolve_api_key_env(row))
    if not api_key:
        raise WorkerContractError("missing OpenAI API key")

    model = str(row.get("model", "")).strip()
    if not model:
        raise WorkerContractError("external_api request row missing model")

    params_in = _as_dict(row.get("model_params"))
    reasoning_effort = str(params_in.get("reasoning_effort", "")).strip()
    text_verbosity = str(params_in.get("text_verbosity", "")).strip()

    system_prompt, user_prompt = _prompt_pair(row)
    client = OpenAI(api_key=api_key, timeout=_provider_timeout(row, "openai"))

    request_kwargs: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
        "temperature": 0.0,
    }
    if reasoning_effort:
        request_kwargs["reasoning"] = {"effort": reasoning_effort}
    if text_verbosity:
        request_kwargs["text"] = {"verbosity": text_verbosity}

    started = time.perf_counter()
    response = _call_openai_with_fallback(client, request_kwargs)
    latency_ms = (time.perf_counter() - started) * 1000.0

    output_text = getattr(response, "output_text", None)
    thinking_text = ""
    if not isinstance(output_text, str) or not output_text.strip():
        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            for part in getattr(item, "content", None) or []:
                v = getattr(part, "text", None)
                if isinstance(v, str):
                    chunks.append(v)
                elif isinstance(v, Mapping):
                    chunks.append(str(v.get("value", v.get("text", "")) or ""))
        output_text = "".join(chunks)
    # Reasoning summaries (rare) come as items with type='reasoning'.
    for item in getattr(response, "output", []) or []:
        itype = getattr(item, "type", None)
        if itype == "reasoning":
            summary = getattr(item, "summary", None)
            if summary:
                for s in summary:
                    v = getattr(s, "text", None)
                    if isinstance(v, str) and v:
                        thinking_text += (("\n" if thinking_text else "") + v)
                    elif isinstance(v, Mapping):
                        chunk = str(v.get("value", v.get("text", "")) or "")
                        if chunk:
                            thinking_text += (("\n" if thinking_text else "") + chunk)

    if not output_text or not output_text.strip():
        raise WorkerContractError("OpenAI response did not contain output_text")

    teacher_label, payload_text = _split_label_and_text(output_text)
    status = "ok" if teacher_label != "invalid" else "filtered"

    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
    reasoning_tokens = 0
    if usage is not None:
        details = getattr(usage, "output_tokens_details", None)
        if details is not None:
            reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
        elif isinstance(usage, Mapping):
            reasoning_tokens = int(
                _as_dict(usage.get("output_tokens_details")).get("reasoning_tokens", 0) or 0
            )

    return {
        "status": status,
        "gold": payload_text if status == "ok" else None,
        "teacher_label": teacher_label,
        "thinking_text": thinking_text,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "reasoning_tokens": reasoning_tokens,
        },
        "latency_ms": round(latency_ms, 3),
        "reason": payload_text if status != "ok" else None,
        "error": None,
    }


def _call_openai_with_fallback(client: Any, request_kwargs: dict[str, Any]) -> Any:
    """Run responses.create, retrying after stripping rejected fields."""

    params = dict(request_kwargs)
    last_err: Exception | None = None
    for _ in range(4):
        try:
            return client.responses.create(**params)
        except Exception as exc:  # pylint: disable=broad-except
            msg = str(exc)
            removed = False
            for key in ("temperature", "top_p", "reasoning", "text"):
                if key in params and (
                    "Unsupported parameter" in msg
                    or "unsupported" in msg.lower()
                    or "Invalid" in msg
                ) and (f"'{key}'" in msg or key in msg):
                    params.pop(key, None)
                    removed = True
            if not removed:
                raise
            last_err = exc
    if last_err is not None:
        raise last_err
    raise WorkerContractError("OpenAI responses.create exhausted retries")


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------


def _anthropic_call(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import anthropic
    except ModuleNotFoundError as exc:
        raise WorkerContractError("anthropic package is required for external_api worker") from exc

    api_key = os.environ.get(_resolve_api_key_env(row))
    if not api_key:
        raise WorkerContractError("missing Anthropic API key")

    model = str(row.get("model", "")).strip()
    if not model:
        raise WorkerContractError("external_api request row missing model")

    params_in = _as_dict(row.get("model_params"))
    thinking_mode = str(params_in.get("thinking_mode", "off")).strip().lower()
    adaptive_effort = str(params_in.get("adaptive_effort", "medium")).strip().lower()
    thinking_display = str(params_in.get("thinking_display", "summarized")).strip().lower()

    max_tokens = int(params_in.get("max_tokens", 8192) or 8192)
    if thinking_mode == "adaptive" and max_tokens < 4096:
        max_tokens = 4096

    system_prompt, user_prompt = _prompt_pair(row)

    client = anthropic.Anthropic(
        api_key=api_key,
        timeout=_provider_timeout(row, "anthropic"),
    )
    request_kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if thinking_mode == "adaptive":
        request_kwargs["thinking"] = {"type": "adaptive", "display": thinking_display}
        request_kwargs["output_config"] = {"effort": adaptive_effort}
        request_kwargs["temperature"] = 1.0
    else:
        request_kwargs["temperature"] = 0.0

    started = time.perf_counter()
    response = _call_anthropic_with_fallback(
        client, request_kwargs, is_adaptive=(thinking_mode == "adaptive")
    )
    latency_ms = (time.perf_counter() - started) * 1000.0

    text_chunks: list[str] = []
    thinking_chunks: list[str] = []
    for block in getattr(response, "content", []) or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_chunks.append(getattr(block, "text", "") or "")
        elif btype == "thinking":
            thinking_chunks.append(getattr(block, "thinking", "") or "")

    output_text = "".join(text_chunks).strip()
    if not output_text:
        raise WorkerContractError("Anthropic response did not contain text content")

    teacher_label, payload_text = _split_label_and_text(output_text)
    status = "ok" if teacher_label != "invalid" else "filtered"

    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0

    return {
        "status": status,
        "gold": payload_text if status == "ok" else None,
        "teacher_label": teacher_label,
        "thinking_text": "\n".join(t for t in thinking_chunks if t),
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            # Anthropic includes thinking tokens inside output_tokens; no separate count.
            "reasoning_tokens": 0,
        },
        "latency_ms": round(latency_ms, 3),
        "reason": payload_text if status != "ok" else None,
        "error": None,
    }


def _call_anthropic_with_fallback(
    client: Any, request_kwargs: dict[str, Any], *, is_adaptive: bool
) -> Any:
    params = dict(request_kwargs)
    last_err: Exception | None = None
    for _ in range(3):
        try:
            return client.messages.create(**params)
        except Exception as exc:  # pylint: disable=broad-except
            msg = str(exc)
            removed = False
            # Strip rejected fields based on the error message.
            for key in ("top_p", "temperature", "output_config", "thinking"):
                if key in params and key in msg and (
                    "400" in msg or "invalid" in msg.lower() or "bad_request" in msg.lower()
                ):
                    if key == "thinking" and is_adaptive:
                        continue  # never strip thinking on adaptive
                    params.pop(key, None)
                    removed = True
            if not removed:
                raise
            last_err = exc
    if last_err is not None:
        raise last_err
    raise WorkerContractError("Anthropic messages.create exhausted retries")


# ---------------------------------------------------------------------------
# Gemini adapter
# ---------------------------------------------------------------------------


def _gemini_call(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise WorkerContractError("requests package is required for Gemini adapter") from exc

    api_key = os.environ.get(_resolve_api_key_env(row))
    if not api_key:
        raise WorkerContractError("missing Gemini API key")

    model = str(row.get("model", "")).strip()
    if not model:
        raise WorkerContractError("external_api request row missing model")

    params_in = _as_dict(row.get("model_params"))
    thinking_mode = str(params_in.get("thinking_mode", "off")).strip().lower()
    thinking_budget = int(params_in.get("thinking_budget", -1) or -1)
    max_tokens = int(params_in.get("max_tokens", 8192) or 8192)

    system_prompt, user_prompt = _prompt_pair(row)
    timeout = _provider_timeout(row, "gemini")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{model}:generateContent?key={api_key}"
    )

    gen_cfg: dict[str, Any] = {
        "temperature": 0.0,
        "maxOutputTokens": max_tokens,
        "responseMimeType": "text/plain",
    }
    if thinking_mode == "dynamic":
        gen_cfg["thinkingConfig"] = {
            "thinkingBudget": thinking_budget,
            "includeThoughts": True,
        }
        if max_tokens < 4096:
            gen_cfg["maxOutputTokens"] = 4096
    else:
        gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": gen_cfg,
    }

    started = time.perf_counter()
    response_data, _ = _call_gemini_with_fallback(url=url, payload=payload, timeout=timeout)
    latency_ms = (time.perf_counter() - started) * 1000.0

    candidates = response_data.get("candidates", []) or []
    text_chunks: list[str] = []
    thinking_chunks: list[str] = []
    for cand in candidates:
        for part in _as_dict(cand.get("content")).get("parts", []) or []:
            text_value = str(part.get("text", "") or "")
            if not text_value:
                continue
            if part.get("thought") is True:
                thinking_chunks.append(text_value)
            else:
                text_chunks.append(text_value)

    output_text = "".join(text_chunks).strip()
    if not output_text:
        reasons = [str(c.get("finishReason", "")) for c in candidates]
        raise WorkerContractError(f"Gemini empty text (finishReason={reasons})")

    teacher_label, payload_text = _split_label_and_text(output_text)
    status = "ok" if teacher_label != "invalid" else "filtered"

    usage_meta = _as_dict(response_data.get("usageMetadata"))
    input_tokens = int(usage_meta.get("promptTokenCount", 0) or 0)
    visible_output_tokens = int(usage_meta.get("candidatesTokenCount", 0) or 0)
    thoughts_tokens = int(usage_meta.get("thoughtsTokenCount", 0) or 0)
    output_tokens = visible_output_tokens + thoughts_tokens

    return {
        "status": status,
        "gold": payload_text if status == "ok" else None,
        "teacher_label": teacher_label,
        "thinking_text": "\n".join(thinking_chunks),
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "reasoning_tokens": thoughts_tokens,
        },
        "latency_ms": round(latency_ms, 3),
        "reason": payload_text if status != "ok" else None,
        "error": None,
    }


def _call_gemini_with_fallback(
    *, url: str, payload: dict[str, Any], timeout: float
) -> tuple[dict[str, Any], int]:
    import requests

    last_err: Exception | None = None
    for attempt in range(5):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            last_err = exc
            time.sleep(min(2 ** attempt, 30))
            continue

        if response.status_code < 300:
            try:
                return response.json(), response.status_code
            except ValueError as exc:
                raise WorkerContractError(f"Gemini returned non-JSON body: {exc}") from exc

        body = response.text[:500]
        last_err = WorkerContractError(f"Gemini API error {response.status_code}: {body}")

        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(2 ** attempt, 30))
            continue
        if response.status_code == 400:
            gc = payload.get("generationConfig")
            removed = False
            if isinstance(gc, dict):
                if "thinking" in body.lower() and "thinkingConfig" in gc:
                    gc.pop("thinkingConfig", None)
                    removed = True
                if (
                    "responseMimeType" in body or "mime" in body.lower()
                ) and "responseMimeType" in gc:
                    gc.pop("responseMimeType", None)
                    removed = True
            if removed:
                continue
        raise last_err

    if last_err is not None:
        raise last_err
    raise WorkerContractError("Gemini retries exhausted")


# ---------------------------------------------------------------------------
# Dispatch & parallel orchestration
# ---------------------------------------------------------------------------


def _dispatch_call(row: Mapping[str, Any]) -> dict[str, Any]:
    provider = str(row.get("provider", "")).strip().lower()
    if provider == "openai":
        return _openai_call(row)
    if provider in {"anthropic", "claude"}:
        return _anthropic_call(row)
    if provider == "gemini":
        return _gemini_call(row)
    raise WorkerContractError(
        f"provider={provider!r} is not implemented in external_api worker"
    )


def _fallback_error_response(row: Mapping[str, Any], message: str) -> dict[str, Any]:
    split_name = row.get("split_name")
    if split_name is not None and not isinstance(split_name, str):
        split_name = str(split_name)
    return {
        "request_id": str(row.get("request_id", "")),
        "status": "failed",
        "gold": None,
        "teacher_label": "runtime_error",
        "thinking_text": "",
        "split_name": split_name,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
        },
        "cost": {"currency": "USD", "estimated": 0.0},
        "latency_ms": 0.0,
        "attempt": 1,
        "reason": message,
        "error": message,
    }


def _wrap_response(row: Mapping[str, Any], call_result: Mapping[str, Any]) -> dict[str, Any]:
    split_name = row.get("split_name")
    if split_name is not None and not isinstance(split_name, str):
        split_name = str(split_name)
    return {
        "request_id": str(row.get("request_id", "")),
        "status": call_result["status"],
        "gold": call_result.get("gold"),
        "teacher_label": call_result["teacher_label"],
        "thinking_text": call_result.get("thinking_text", "") or "",
        "split_name": split_name,
        "usage": call_result["usage"],
        "cost": {"currency": "USD", "estimated": 0.0},
        "latency_ms": call_result["latency_ms"],
        "attempt": 1,
        "reason": call_result.get("reason"),
        "error": call_result.get("error"),
    }


def _load_concurrency(rows: list[dict[str, Any]]) -> tuple[int, dict[str, int], float]:
    if not rows:
        return 4, dict(_PROVIDER_CONCURRENCY_DEFAULT), 0.0
    first_cfg = _as_dict(_as_dict(rows[0].get("runtime_config")).get("external_api"))
    concurrency_cfg = _as_dict(first_cfg.get("concurrency"))
    try:
        max_workers = max(1, int(concurrency_cfg.get("max_workers", 12)))
    except (TypeError, ValueError):
        max_workers = 12
    per_provider_raw = _as_dict(concurrency_cfg.get("per_provider"))
    per_provider: dict[str, int] = {}
    for key, value in _PROVIDER_CONCURRENCY_DEFAULT.items():
        try:
            per_provider[key] = max(1, int(per_provider_raw.get(key, value)))
        except (TypeError, ValueError):
            per_provider[key] = value
    try:
        min_interval = max(0.0, float(concurrency_cfg.get("min_request_interval_sec", 0.0)))
    except (TypeError, ValueError):
        min_interval = 0.0
    return max_workers, per_provider, min_interval


def _progress_emit(done: int, total: int, *, force: bool = False) -> None:
    """Emit a one-line progress update to stderr (carriage-return updated)."""

    if total <= 0:
        return
    if not force and done not in (1, total) and done % max(1, total // 100) != 0:
        return
    pct = (done / total) * 100.0
    bar_width = 24
    filled = int(math.floor(bar_width * done / total))
    bar = "#" * filled + "-" * (bar_width - filled)
    end = "\n" if done >= total else "\r"
    sys.stderr.write(f"external_api [{bar}] {done}/{total} ({pct:.1f}%){end}")
    sys.stderr.flush()


def main(argv: list[str] | None = None) -> int:
    args = parse_worker_args(description="Real external API worker", argv=argv)

    requests_in = [dict(row) for row in read_jsonl(args.input_path)]
    schema = validate_phase_request_rows(requests_in, args=args, context="external_api")

    max_workers, per_provider, min_interval = _load_concurrency(requests_in)
    providers_seen = {str(row.get("provider", "openai")).lower() for row in requests_in}
    sems = {
        provider: threading.Semaphore(per_provider.get(provider, 4))
        for provider in providers_seen
    }
    provider_locks = {provider: threading.Lock() for provider in providers_seen}
    next_provider_time = {provider: 0.0 for provider in providers_seen}

    def _run_one(idx: int, row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        provider = str(row.get("provider", "openai")).lower()
        sem = sems.get(provider) or threading.Semaphore(4)
        with sem:
            if min_interval > 0 and provider in provider_locks:
                with provider_locks[provider]:
                    wait_for = next_provider_time[provider] - time.time()
                    if wait_for > 0:
                        time.sleep(wait_for)
                    next_provider_time[provider] = time.time() + min_interval
            try:
                call_result = _dispatch_call(row)
                response = _wrap_response(row, call_result)
            except Exception as exc:  # pylint: disable=broad-except
                response = _fallback_error_response(row, str(exc))
        return idx, response

    total = len(requests_in)
    responses: list[dict[str, Any] | None] = [None] * total
    done = 0
    _progress_emit(done, total, force=True)

    if total:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_run_one, idx, row) for idx, row in enumerate(requests_in)
            ]
            for fut in as_completed(futures):
                idx, response = fut.result()
                responses[idx] = response
                done += 1
                _progress_emit(done, total)

    final_responses: list[dict[str, Any]] = []
    for idx, response in enumerate(responses):
        if response is None:
            response = _fallback_error_response(requests_in[idx], "no response")
        final_responses.append(response)

    validate_phase_response_rows(final_responses, schema=schema, context="external_api")
    write_jsonl(args.output_path, final_responses, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
