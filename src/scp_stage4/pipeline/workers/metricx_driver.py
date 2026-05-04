"""MetricX subprocess driver for QE worker.

Reads JSONL rows with:
  {"source": "...", "hypothesis": "...", "reference": "...?"}

Writes JSONL rows with:
  {"prediction": float}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from scp_stage4.data import read_jsonl, write_jsonl

_METRICX_REGRESSION_TOKEN_ID = 250089


def _format_input(row: dict[str, Any], mode: str) -> str:
    source = str(row.get("source", ""))
    hypothesis = str(row.get("hypothesis", ""))
    if mode == "qe":
        return f"source: {source} candidate: {hypothesis}"
    reference = str(row.get("reference", ""))
    return f"source: {source} candidate: {hypothesis} reference: {reference}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MetricX subprocess driver")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--tokenizer", default="google/mt5-xl")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=1536)
    parser.add_argument("--qe", action="store_true")
    args = parser.parse_args(argv)

    mode = "qe" if args.qe else "ref"
    rows = [dict(row) for row in read_jsonl(Path(args.input_file))]
    if not rows:
        write_jsonl(Path(args.output_file), [], ensure_ascii=False)
        return 0

    batch_size = max(1, int(args.batch_size))
    max_input_length = max(1, int(args.max_input_length))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
    )
    model.to(device)
    model.eval()

    formatted = [_format_input(row, mode=mode) for row in rows]
    predictions: list[float] = []
    for start in range(0, len(formatted), batch_size):
        chunk = formatted[start : start + batch_size]
        encoded = tokenizer(
            chunk,
            max_length=max_input_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        decoder_input_ids = torch.zeros(
            (input_ids.shape[0], 1),
            dtype=torch.long,
            device=device,
        )
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            )
            scores = (
                output.logits[:, 0, _METRICX_REGRESSION_TOKEN_ID]
                .float()
                .clamp(0.0, 25.0)
                .tolist()
            )
            predictions.extend(float(score) for score in scores)

    write_jsonl(
        Path(args.output_file),
        [{"prediction": score} for score in predictions],
        ensure_ascii=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

