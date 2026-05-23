#!/usr/bin/env python3
"""Translate ad hoc English text with a local SCP checkpoint or base model."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from scp_stage4.config.loader import compose_config
from scp_stage4.data import read_jsonl, write_jsonl


def _get_by_dotpath(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    cursor: Any = cfg
    for part in key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _load_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return str(args.text)
    if args.input_file is not None:
        return Path(args.input_file).read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("provide --text, --input-file, or stdin")


def _build_decoding(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    decoding = dict(_get_by_dotpath(cfg, "inference.eval", {}) or {})
    if args.max_new_tokens is not None:
        decoding["max_new_tokens"] = int(args.max_new_tokens)
    if args.sample:
        decoding["do_sample"] = True
        decoding["temperature"] = float(args.temperature)
        decoding["top_p"] = float(args.top_p)
    else:
        decoding["do_sample"] = False
        decoding["temperature"] = 0.0
        decoding["top_p"] = None
    return decoding


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="Local full-weight checkpoint or LoRA adapter path")
    parser.add_argument("--base-model-only", action="store_true", help="Load config model.name without a checkpoint")
    parser.add_argument("--config", default="configs/scp_stage4_real_1gpu_greedy_eval.yaml")
    parser.add_argument("--text", default=None)
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--output", default=None, help="Optional JSONL output path")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--sample", action="store_true", help="Use sampling instead of greedy decoding")
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--run-id", default="manual_translate")
    args, overrides = parser.parse_known_args(argv)

    checkpoint: Path | None = None
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.exists():
            raise SystemExit(f"checkpoint path not found: {checkpoint}")
    elif not args.base_model_only:
        raise SystemExit("provide --checkpoint or --base-model-only")

    source = _load_text(args).strip()
    if not source:
        raise SystemExit("source text is empty")

    cfg = compose_config(args.config, overrides=overrides)
    request = {
        "id": f"{args.run_id}/manual/000/ood",
        "run_id": args.run_id,
        "subset_idx": 0,
        "row_id": "manual_000",
        "order_idx": 0,
        "q_tag": "ood",
        "source": source,
        "metadata": {},
        "decoding": _build_decoding(cfg, args),
        "runtime_config": {
            "model": _get_by_dotpath(cfg, "model", {}),
            "inference": _get_by_dotpath(cfg, "inference", {}),
            "data_length": _get_by_dotpath(cfg, "data.length", {}),
            "prompts": _get_by_dotpath(cfg, "prompts", {}),
        },
    }
    if checkpoint is not None:
        request["base_checkpoint"] = str(checkpoint)

    with tempfile.TemporaryDirectory(prefix="scp_translate_") as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / "input.jsonl"
        output_path = tmp_dir / "output.jsonl"
        write_jsonl(input_path, [request], ensure_ascii=False)
        cmd = [
            sys.executable,
            "-m",
            "scp_stage4.pipeline.workers.vllm_inference_worker",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--section",
            "inference",
            "--phase",
            "infer-ood",
            "--run-id",
            args.run_id,
            "--subset-idx",
            "0",
        ]
        subprocess.run(cmd, check=True)
        rows = read_jsonl(output_path)

    if not rows:
        raise SystemExit("worker produced no output rows")
    row = rows[0]
    if row.get("status") != "ok":
        raise SystemExit(f"translation failed: {row.get('error')}")

    result = {
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "model": str(_get_by_dotpath(cfg, "model.name", "")),
        "base_model_only": checkpoint is None,
        "source": source,
        "translation": str(row.get("mt", "")),
    }
    if args.output:
        write_jsonl(Path(args.output), [result], ensure_ascii=False)
    print(result["translation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
