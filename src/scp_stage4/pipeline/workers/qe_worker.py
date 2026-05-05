"""Real QE worker for subprocess runtime.

Supports:
- metricx24 via `python -m metricx24.predict`
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


def _metricx24_scores(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    tokenizer_name: str,
    batch_size: int,
    max_input_length: int,
) -> list[float]:
    metricx_python = _resolve_isolation_python("METRICX_PYTHON")
    with tempfile.TemporaryDirectory(prefix="scp_qe_metricx_") as tmpdir:
        tmp = Path(tmpdir)
        input_path = tmp / "input.jsonl"
        output_path = tmp / "output.jsonl"
        payload = [
            {
                "source": str(row.get("src", "")),
                "hypothesis": str(row.get("mt", "")),
                "reference": "",
            }
            for row in rows
        ]
        write_jsonl(input_path, payload, ensure_ascii=False)
        cmd = [
            metricx_python,
            "-m",
            "metricx24.predict",
            "--tokenizer",
            tokenizer_name,
            "--model_name_or_path",
            model_name,
            "--max_input_length",
            str(max_input_length),
            "--batch_size",
            str(batch_size),
            "--input_file",
            str(input_path),
            "--output_file",
            str(output_path),
            "--qe",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            detail = (result.stderr or "").strip() or (result.stdout or "").strip() or "no output"
            raise WorkerContractError(f"metricx24.predict failed: {detail}")

        out_rows = read_jsonl(output_path)
        if len(out_rows) != len(rows):
            raise WorkerContractError(
                f"metricx24 output row mismatch: expected={len(rows)}, got={len(out_rows)}"
            )

        scores: list[float] = []
        for row in out_rows:
            pred = row.get("prediction")
            if isinstance(pred, bool) or not isinstance(pred, (int, float)):
                raise WorkerContractError("metricx24 output row missing numeric prediction")
            scores.append(float(pred))
        return scores


def _comet_scores(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    batch_size: int,
) -> list[float]:
    comet_python = _resolve_isolation_python("COMET_PYTHON")
    if comet_python == sys.executable:
        try:
            from comet import download_model, load_from_checkpoint
        except ModuleNotFoundError as exc:
            raise WorkerContractError(
                "comet package is required for qe.primary.backend=comet_kiwi; "
                "set COMET_PYTHON to a venv with unbabel-comet installed"
            ) from exc

        data = [{"src": str(row.get("src", "")), "mt": str(row.get("mt", ""))} for row in rows]
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
        data = [{"src": str(row.get("src", "")), "mt": str(row.get("mt", ""))} for row in rows]
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
            [comet_python, "-c", script], capture_output=True, text=True
        )
        if result.returncode != 0:
            detail = (result.stderr or "").strip() or (result.stdout or "").strip() or "no output"
            raise WorkerContractError(f"COMET subprocess failed: {detail}")
        if not output_path.exists():
            raise WorkerContractError("COMET subprocess did not produce output")
        values = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(values, list) or len(values) != len(rows):
            raise WorkerContractError("COMET subprocess did not return per-row scores")
        return [float(v) for v in values]


def _score_rows(rows: list[dict[str, Any]]) -> tuple[list[float], str]:
    runtime_cfg = _as_dict(rows[0].get("runtime_config"))
    qe_primary = _as_dict(runtime_cfg.get("qe_primary"))

    backend = str(rows[0].get("backend", qe_primary.get("backend", "heuristic")))
    model_name = str(qe_primary.get("model_name", "")).strip()
    tokenizer_name = str(qe_primary.get("tokenizer_name", "")).strip()
    batch_size = int(qe_primary.get("batch_size", 8) or 8)
    max_input_length = int(qe_primary.get("max_input_length", 1536) or 1536)
    if batch_size <= 0:
        batch_size = 1
    if max_input_length <= 0:
        max_input_length = 1536

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
            ),
            model_name,
        )

    if backend == "comet_kiwi":
        if not model_name:
            raise WorkerContractError("qe.primary.model_name is required for comet_kiwi backend")
        return (_comet_scores(rows, model_name=model_name, batch_size=batch_size), model_name)

    if backend == "heuristic":
        return (
            [_heuristic_score(str(row.get("src", "")), str(row.get("mt", ""))) for row in rows],
            "heuristic/local",
        )

    raise WorkerContractError(
        f"unsupported QE backend={backend!r}. Supported: metricx24, comet_kiwi, heuristic"
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
    raise SystemExit(main())
