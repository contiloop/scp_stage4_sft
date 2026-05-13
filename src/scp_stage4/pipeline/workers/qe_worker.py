"""Real QE worker for subprocess runtime.

Supports:
- metricx24 via inline driver script (no metricx24 package needed)
- metricx24_ref (reference-based MetricX scoring for OOD eval)
- BLEU / chrF via sacrebleu sentence-level metrics
- comet_kiwi via COMET python package
- heuristic fallback for lightweight local runs
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
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


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    normalized = text.replace(" ", "")
    if len(normalized) < n:
        return {normalized} if normalized else set()
    return {normalized[i : i + n] for i in range(len(normalized) - n + 1)}


def _heuristic_score(src: str, mt: str) -> float:
    src_grams = _char_ngrams(src.lower())
    mt_grams = _char_ngrams(mt.lower())
    if not src_grams or not mt_grams:
        return 0.0
    overlap = len(src_grams & mt_grams) / max(len(src_grams), 1)
    length_ratio = min(len(mt), len(src)) / max(len(mt), len(src), 1)
    return float(round(max(0.0, min(1.0, 0.7 * overlap + 0.3 * length_ratio)), 6))


def _resolve_isolation_python(env_var: str) -> str:
    value = os.environ.get(env_var, "").strip()
    if value and Path(value).exists():
        return value
    return sys.executable


_METRICX_DRIVER_SCRIPT = """\
import json, sys, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

args = json.loads(sys.stdin.read())
model_name = args["model_name"]
tokenizer_name = args.get("tokenizer_name") or "google/mt5-xl"
batch_size = int(args.get("batch_size", 8))
max_input_length = int(args.get("max_input_length", 1536))
payload = args["payload"]

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available in QE venv - refusing CPU fallback")
device = "cuda"
print(f"[metricx-driver] loading {model_name} on {device}", file=sys.stderr)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype="auto")
model.to(device)
model.eval()
print(f"[metricx-driver] model loaded, scoring {len(payload)} rows", file=sys.stderr)

formatted = []
for r in payload:
    src = r.get("src", "")
    mt = r.get("mt", "")
    ref = r.get("ref", "")
    text = f"source: {src} candidate: {mt}"
    if ref:
        text += f" reference: {ref}"
    formatted.append(text)
scores = []
total_batches = (len(formatted) + batch_size - 1) // batch_size
for batch_idx, start in enumerate(range(0, len(formatted), batch_size)):
    chunk = formatted[start:start + batch_size]
    enc = tokenizer(
        chunk,
        max_length=max_input_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    decoder_input_ids = torch.zeros((input_ids.shape[0], 1), dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            decoder_input_ids=decoder_input_ids,
        )
        batch_scores = out.logits[:, 0, 250089].float().clamp(0.0, 25.0).tolist()
        scores.extend(float(x) for x in batch_scores)
    if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
        print(f"[metricx-driver] {batch_idx+1}/{total_batches} batches ({len(scores)}/{len(formatted)} rows)", file=sys.stderr)

print(json.dumps({"model_name": model_name, "scores": scores}))
"""


def _metricx24_scores(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    tokenizer_name: str,
    batch_size: int,
    max_input_length: int,
    include_reference: bool = False,
) -> list[float]:
    metricx_python = _resolve_isolation_python("METRICX_PYTHON")
    payload: list[dict[str, str]] = []
    for row in rows:
        item = {"src": str(row.get("src", "")), "mt": str(row.get("mt", ""))}
        if include_reference:
            ref = str(row.get("ref", "")).strip()
            if not ref:
                raise WorkerContractError(
                    "metricx24_ref requires non-empty ref field for every row"
                )
            item["ref"] = ref
        payload.append(item)
    args = json.dumps({
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
        "batch_size": batch_size,
        "max_input_length": max_input_length,
        "payload": payload,
    })

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(_METRICX_DRIVER_SCRIPT)
        driver_path = fh.name

    try:
        env = os.environ.copy()
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        result = subprocess.run(
            [metricx_python, driver_path],
            input=args,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            env=env,
        )
    finally:
        try:
            os.unlink(driver_path)
        except OSError:
            pass

    if result.returncode != 0:
        detail = (result.stderr or "").strip() or (result.stdout or "").strip() or "no output"
        raise WorkerContractError(f"metricx24 driver failed: {detail}")

    try:
        out = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise WorkerContractError(f"metricx24 driver output parse error: {exc}")

    scores = out.get("scores", [])
    if len(scores) != len(rows):
        raise WorkerContractError(
            f"metricx24 output row mismatch: expected={len(rows)}, got={len(scores)}"
        )
    return [float(s) for s in scores]


_SACREBLEU_DRIVER_SCRIPT = """\
import json, sys

args = json.loads(sys.stdin.read())
metric = args["metric"]
payload = args["payload"]
settings = args.get("settings", {}) or {}
scores = []

if metric == "bleu":
    from sacrebleu.metrics import BLEU

    scorer = BLEU(
        effective_order=bool(settings.get("effective_order", True)),
        smooth_method=str(settings.get("smooth_method", "exp")),
    )
    for row in payload:
        mt = str(row.get("mt", ""))
        ref = str(row.get("ref", ""))
        scores.append(float(scorer.sentence_score(mt, [ref]).score))
elif metric == "chrf":
    from sacrebleu.metrics import CHRF

    raw_word_order = settings.get("word_order", 2)
    try:
        word_order = int(raw_word_order)
    except Exception:
        word_order = 2
    if word_order < 0:
        word_order = 2
    scorer = CHRF(word_order=word_order)
    for row in payload:
        mt = str(row.get("mt", ""))
        ref = str(row.get("ref", ""))
        scores.append(float(scorer.sentence_score(mt, [ref]).score))
else:
    raise RuntimeError(f"unsupported metric: {metric}")

print(json.dumps({"scores": scores}))
"""


def _sacrebleu_scores_subprocess(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    metric_settings: Mapping[str, Any] | None,
) -> list[float]:
    metricx_python = _resolve_isolation_python("METRICX_PYTHON")
    payload: list[dict[str, str]] = []
    for row in rows:
        ref = str(row.get("ref", "")).strip()
        if not ref:
            raise WorkerContractError(f"{metric} backend requires non-empty ref field")
        payload.append({"mt": str(row.get("mt", "")), "ref": ref})

    args = json.dumps(
        {
            "metric": metric,
            "payload": payload,
            "settings": dict(metric_settings) if isinstance(metric_settings, Mapping) else {},
        }
    )

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(_SACREBLEU_DRIVER_SCRIPT)
        driver_path = fh.name
    try:
        result = subprocess.run(
            [metricx_python, driver_path],
            input=args,
            capture_output=True,
            text=True,
        )
    finally:
        try:
            os.unlink(driver_path)
        except OSError:
            pass

    if result.returncode != 0:
        detail = (result.stderr or "").strip() or (result.stdout or "").strip() or "no output"
        raise WorkerContractError(f"sacrebleu {metric} driver failed: {detail}")

    try:
        out = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise WorkerContractError(f"sacrebleu {metric} driver output parse error: {exc}")
    scores = out.get("scores", [])
    if not isinstance(scores, list) or len(scores) != len(rows):
        raise WorkerContractError(
            f"sacrebleu {metric} output row mismatch: expected={len(rows)}, got={len(scores)}"
        )
    return [float(score) for score in scores]


def _bleu_scores(
    rows: list[dict[str, Any]],
    *,
    metric_settings: Mapping[str, Any] | None,
) -> list[float]:
    metricx_python = _resolve_isolation_python("METRICX_PYTHON")
    if metricx_python != sys.executable:
        return _sacrebleu_scores_subprocess(
            rows,
            metric="bleu",
            metric_settings=metric_settings,
        )
    try:
        from sacrebleu.metrics import BLEU
    except ModuleNotFoundError as exc:
        raise WorkerContractError(
            "sacrebleu is required for BLEU backend in QE worker. "
            "Install it in the current Python, or set METRICX_PYTHON to a venv with sacrebleu."
        ) from exc

    settings = dict(metric_settings) if isinstance(metric_settings, Mapping) else {}
    effective_order = bool(settings.get("effective_order", True))
    smooth_method = str(settings.get("smooth_method", "exp"))
    metric = BLEU(effective_order=effective_order, smooth_method=smooth_method)

    out: list[float] = []
    for row in rows:
        mt = str(row.get("mt", ""))
        ref = str(row.get("ref", "")).strip()
        if not ref:
            raise WorkerContractError("BLEU backend requires non-empty ref field")
        out.append(float(metric.sentence_score(mt, [ref]).score))
    return out


def _chrf_scores(
    rows: list[dict[str, Any]],
    *,
    metric_settings: Mapping[str, Any] | None,
) -> list[float]:
    metricx_python = _resolve_isolation_python("METRICX_PYTHON")
    if metricx_python != sys.executable:
        return _sacrebleu_scores_subprocess(
            rows,
            metric="chrf",
            metric_settings=metric_settings,
        )
    try:
        from sacrebleu.metrics import CHRF
    except ModuleNotFoundError as exc:
        raise WorkerContractError(
            "sacrebleu is required for chrF backend in QE worker. "
            "Install it in the current Python, or set METRICX_PYTHON to a venv with sacrebleu."
        ) from exc

    settings = dict(metric_settings) if isinstance(metric_settings, Mapping) else {}
    word_order = int(settings.get("word_order", 2) or 2)
    if word_order < 0:
        word_order = 2
    metric = CHRF(word_order=word_order)

    out: list[float] = []
    for row in rows:
        mt = str(row.get("mt", ""))
        ref = str(row.get("ref", "")).strip()
        if not ref:
            raise WorkerContractError("chrF backend requires non-empty ref field")
        out.append(float(metric.sentence_score(mt, [ref]).score))
    return out


def _comet_scores(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    batch_size: int,
    include_reference: bool = False,
) -> list[float]:
    comet_python = _resolve_isolation_python("COMET_PYTHON")

    def _build_data() -> list[dict[str, str]]:
        data: list[dict[str, str]] = []
        for row in rows:
            item: dict[str, str] = {"src": str(row.get("src", "")), "mt": str(row.get("mt", ""))}
            if include_reference:
                ref = str(row.get("ref", "")).strip()
                if not ref:
                    raise WorkerContractError(
                        "COMET reference mode requires non-empty ref field for every row"
                    )
                item["ref"] = ref
            data.append(item)
        return data

    if comet_python == sys.executable:
        try:
            from comet import download_model, load_from_checkpoint
        except ModuleNotFoundError as exc:
            raise WorkerContractError(
                "comet package is required; "
                "set COMET_PYTHON to a venv with unbabel-comet installed"
            ) from exc

        data = _build_data()
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)
        try:
            import torch

            gpus = 1 if torch.cuda.is_available() else 0
        except Exception:
            gpus = 0

        pred = model.predict(data, batch_size=batch_size, gpus=gpus)
        if isinstance(pred, Mapping):
            values = pred.get("scores")
        else:
            values = getattr(pred, "scores", None)
        if not isinstance(values, list) or len(values) != len(rows):
            raise WorkerContractError("COMET prediction did not return per-row scores")
        return [float(value) for value in values]

    with tempfile.TemporaryDirectory(prefix="scp_qe_comet_") as tmpdir:
        tmp = Path(tmpdir)
        input_path = tmp / "input.json"
        output_path = tmp / "output.json"
        data = _build_data()
        input_path.write_text(json.dumps(data), encoding="utf-8")

        script = (
            "import json, sys\n"
            "from comet import download_model, load_from_checkpoint\n"
            "try:\n"
            "    import torch; gpus = 1 if torch.cuda.is_available() else 0\n"
            "except Exception:\n"
            "    gpus = 0\n"
            f"data = json.loads(open({str(input_path)!r}).read())\n"
            f"model = load_from_checkpoint(download_model({model_name!r}))\n"
            f"pred = model.predict(data, batch_size={batch_size}, gpus=gpus)\n"
            "scores = pred.get('scores') if isinstance(pred, dict) else getattr(pred, 'scores', None)\n"
            f"open({str(output_path)!r}, 'w').write(json.dumps(scores))\n"
        )
        result = subprocess.run(
            [comet_python, "-c", script], stdout=subprocess.PIPE, stderr=None, text=True
        )
        if result.returncode != 0:
            detail = (result.stdout or "").strip() or "no output (check stderr above)"
            raise WorkerContractError(f"COMET subprocess failed: {detail}")
        if not output_path.exists():
            raise WorkerContractError("COMET subprocess did not produce output")
        values = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(values, list) or len(values) != len(rows):
            raise WorkerContractError("COMET subprocess did not return per-row scores")
        return [float(v) for v in values]


def _backend_metric_settings(
    metric_settings: Mapping[str, Any],
    *,
    backend: str,
) -> dict[str, Any]:
    key_candidates = {
        "metricx24": ("metricx24",),
        "metricx24_ref": ("metricx24_ref",),
        "bleu": ("BLEU", "bleu"),
        "chrf": ("chrF", "chrf"),
        "comet_kiwi": ("comet_kiwi", "cometkiwi"),
        "xcomet": ("xcomet",),
    }.get(backend, ())
    for key in key_candidates:
        scoped = _as_dict(metric_settings.get(key))
        if scoped:
            return scoped
    return {}


def _as_positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except Exception:
        return default
    if parsed <= 0:
        return default
    return parsed


def _score_rows(rows: list[dict[str, Any]]) -> tuple[list[float], str]:
    runtime_cfg = _as_dict(rows[0].get("runtime_config"))
    qe_primary = _as_dict(runtime_cfg.get("qe_primary"))
    metric_settings = _as_dict(runtime_cfg.get("metric_settings"))

    backend_raw = str(rows[0].get("backend", qe_primary.get("backend", "heuristic")))
    backend = backend_raw.strip().lower()
    backend_settings = _backend_metric_settings(metric_settings, backend=backend)
    model_name = str(qe_primary.get("model_name", "")).strip()
    tokenizer_name = str(qe_primary.get("tokenizer_name", "")).strip()
    batch_size = _as_positive_int(qe_primary.get("batch_size", 8), 8)
    max_input_length = _as_positive_int(qe_primary.get("max_input_length", 1536), 1536)

    model_name_override = backend_settings.get("model_name")
    if isinstance(model_name_override, str) and model_name_override.strip():
        model_name = model_name_override.strip()
    tokenizer_name_override = backend_settings.get("tokenizer_name")
    if isinstance(tokenizer_name_override, str) and tokenizer_name_override.strip():
        tokenizer_name = tokenizer_name_override.strip()
    batch_size = _as_positive_int(backend_settings.get("batch_size", batch_size), batch_size)
    max_input_length = _as_positive_int(
        backend_settings.get("max_input_length", max_input_length),
        max_input_length,
    )

    if backend == "metricx24":
        if not model_name:
            raise WorkerContractError("qe.primary.model_name is required for metricx24 backend")
        if not tokenizer_name:
            raise WorkerContractError("qe.primary.tokenizer_name is required for metricx24 backend")
        return (
            _metricx24_scores(
                rows,
                model_name=model_name,
                tokenizer_name=tokenizer_name,
                batch_size=batch_size,
                max_input_length=max_input_length,
                include_reference=False,
            ),
            model_name,
        )

    if backend == "metricx24_ref":
        if not model_name:
            raise WorkerContractError("qe.primary.model_name is required for metricx24_ref backend")
        if not tokenizer_name:
            raise WorkerContractError(
                "qe.primary.tokenizer_name is required for metricx24_ref backend"
            )
        return (
            _metricx24_scores(
                rows,
                model_name=model_name,
                tokenizer_name=tokenizer_name,
                batch_size=batch_size,
                max_input_length=max_input_length,
                include_reference=True,
            ),
            model_name,
        )

    if backend == "bleu":
        return (_bleu_scores(rows, metric_settings=backend_settings), "sacrebleu/BLEU")

    if backend == "chrf":
        return (_chrf_scores(rows, metric_settings=backend_settings), "sacrebleu/chrF")

    if backend == "comet_kiwi":
        if not model_name:
            raise WorkerContractError("qe.primary.model_name is required for comet_kiwi backend")
        return (_comet_scores(rows, model_name=model_name, batch_size=batch_size), model_name)

    if backend == "xcomet":
        if not model_name:
            raise WorkerContractError("qe.primary.model_name is required for xcomet backend")
        return (
            _comet_scores(rows, model_name=model_name, batch_size=batch_size, include_reference=True),
            model_name,
        )

    if backend == "heuristic":
        return (
            [_heuristic_score(str(row.get("src", "")), str(row.get("mt", ""))) for row in rows],
            "heuristic/local",
        )

    raise WorkerContractError(
        "unsupported QE backend="
        f"{backend_raw!r}. Supported: metricx24, metricx24_ref, BLEU, chrF, comet_kiwi, xcomet, heuristic"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_worker_args(description="Real QE worker", argv=argv)

    requests = [dict(row) for row in read_jsonl(args.input_path)]
    schema = validate_phase_request_rows(requests, args=args, context="qe")
    if not requests:
        write_jsonl(args.output_path, [], ensure_ascii=False)
        return 0

    started = time.perf_counter()
    responses: list[dict[str, Any]] = []
    try:
        scores, resolved_model = _score_rows(requests)
        elapsed_ms = max(1.0, (time.perf_counter() - started) * 1000.0)
        per_row_ms = elapsed_ms / max(len(scores), 1)
        for request, score in zip(requests, scores):
            responses.append(
                {
                    "id": str(request.get("id", "")),
                    "score": float(score),
                    "backend": str(request.get("backend", "unknown")),
                    "model_name": resolved_model,
                    "runtime_ms": round(per_row_ms, 3),
                    "status": "ok",
                    "error": None,
                }
            )
    except Exception as exc:
        error_text = str(exc)
        for request in requests:
            responses.append(
                {
                    "id": str(request.get("id", "")),
                    "score": 0.0,
                    "backend": str(request.get("backend", "unknown")),
                    "model_name": "unresolved",
                    "runtime_ms": None,
                    "status": "failed",
                    "error": error_text,
                }
            )

    validate_phase_response_rows(responses, schema=schema, context="qe")
    write_jsonl(args.output_path, responses, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    code = main()
    os._exit(code)
