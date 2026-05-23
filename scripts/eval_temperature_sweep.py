#!/usr/bin/env python3
"""Run OOD eval across model checkpoints and decoding temperatures."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalModel:
    name: str
    subset_idx: int
    checkpoint: Path | None


DEFAULT_MODELS = {
    "qwen35_it": EvalModel("qwen35_it", 0, None),
    "017": EvalModel(
        "subset_017",
        17,
        Path("artifacts/runs/greedy_reeval_main_001/subsets/subset_017/train_final/full_weight_model"),
    ),
    "032": EvalModel(
        "subset_032",
        32,
        Path("artifacts/runs/greedy_reeval_main_001/subsets/subset_032/train_final/full_weight_model"),
    ),
    "034": EvalModel(
        "subset_034",
        34,
        Path(
            "artifacts/runs/greedy_reeval_from032_absrel_no_claude/"
            "subsets/subset_034/train_final/full_weight_model"
        ),
    ),
}


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _temperature_label(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return "t" + text.replace("-", "m").replace(".", "p")


def _write_latest_pointer(
    *,
    run_root: Path,
    run_id: str,
    model: EvalModel,
    temperature: float,
) -> None:
    latest_path = run_root / "checkpoints" / "latest.json"
    if model.checkpoint is None:
        if latest_path.exists():
            latest_path.unlink()
        return

    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(
        json.dumps(
            {
                "checkpoint_path": str(model.checkpoint),
                "run_id": run_id,
                "source": "eval_temperature_sweep",
                "status": "ok",
                "subset_idx": model.subset_idx,
                "temperature": temperature,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _summary_path(run_root: Path, subset_idx: int, dataset: str = "ood_test") -> Path:
    return run_root / "eval" / dataset / f"subset_{subset_idx:03d}.summary.json"


def _load_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _parse_models(raw_models: list[str]) -> list[EvalModel]:
    models: list[EvalModel] = []
    for raw in raw_models:
        key = raw.strip()
        if not key:
            continue
        if key in DEFAULT_MODELS:
            models.append(DEFAULT_MODELS[key])
            continue
        if "=" not in key:
            valid = ", ".join(sorted(DEFAULT_MODELS))
            raise SystemExit(f"unknown model {key!r}; use one of {valid} or name=SUBSET:PATH")
        name, spec = key.split("=", 1)
        if ":" not in spec:
            raise SystemExit(f"custom model spec must be name=SUBSET:PATH, got {raw!r}")
        subset_text, checkpoint_text = spec.split(":", 1)
        models.append(EvalModel(name.strip(), int(subset_text), Path(checkpoint_text)))
    if not models:
        raise SystemExit("at least one model is required")
    return models


def _validate_model_paths(models: list[EvalModel]) -> None:
    missing = [
        str(model.checkpoint)
        for model in models
        if model.checkpoint is not None and not model.checkpoint.exists()
    ]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(
            "missing checkpoint path(s). Run the greedy re-eval restore first or pass a custom model spec:\n"
            f"{joined}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scp_stage4_real_1gpu_greedy_eval.yaml")
    parser.add_argument("--run-prefix", default="temp_sweep")
    parser.add_argument("--models", nargs="+", default=["qwen35_it", "017", "032", "034"])
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 0.3, 0.7, 1.1])
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--run-root-base", default="artifacts/runs")
    parser.add_argument("--clean", action="store_true", help="Remove each sweep run directory before eval")
    parser.add_argument("--skip-existing", action="store_true", help="Skip when the summary artifact already exists")
    args, overrides = parser.parse_known_args(argv)

    models = _parse_models(args.models)
    _validate_model_paths(models)

    index_rows: list[dict[str, object]] = []
    run_root_base = Path(args.run_root_base)

    for model in models:
        for temperature in args.temperatures:
            temp_label = _temperature_label(temperature)
            run_id = f"{args.run_prefix}_{model.name}_{temp_label}"
            run_root = run_root_base / run_id
            summary_path = _summary_path(run_root, model.subset_idx)

            if args.skip_existing and summary_path.exists():
                print(f"[temp-sweep] skip existing {run_id}", flush=True)
                summary = _load_summary(summary_path)
            else:
                if args.clean and run_root.exists():
                    shutil.rmtree(run_root)
                _write_latest_pointer(
                    run_root=run_root,
                    run_id=run_id,
                    model=model,
                    temperature=temperature,
                )
                decoding_overrides = [
                    "inference.eval.do_sample=false",
                    "inference.eval.temperature=0.0",
                    "inference.eval.top_p=null",
                ]
                if temperature > 0:
                    decoding_overrides = [
                        "inference.eval.do_sample=true",
                        f"inference.eval.temperature={temperature}",
                        f"inference.eval.top_p={args.top_p}",
                    ]
                _run(
                    [
                        sys.executable,
                        "-m",
                        "scp_stage4.pipeline.step_subset",
                        "eval-ood",
                        "--config",
                        args.config,
                        "--run-id",
                        run_id,
                        "--subset-idx",
                        str(model.subset_idx),
                        *decoding_overrides,
                        *overrides,
                    ]
                )
                summary = _load_summary(summary_path)

            index_rows.append(
                {
                    "run_id": run_id,
                    "model": model.name,
                    "checkpoint": str(model.checkpoint) if model.checkpoint is not None else None,
                    "subset_idx": model.subset_idx,
                    "temperature": temperature,
                    "top_p": args.top_p if temperature > 0 else None,
                    "do_sample": temperature > 0,
                    "summary_path": str(summary_path),
                    "rows_path": str(run_root / "eval" / "ood_test" / f"subset_{model.subset_idx:03d}.rows.jsonl"),
                    "metricx24_ref_quality_mean": summary.get("metricx24_ref_quality_mean"),
                    "metricx24_ref_raw_error_mean": summary.get("metricx24_ref_raw_error_mean"),
                    "bleu_mean": summary.get("bleu_mean"),
                    "chrf_mean": summary.get("chrf_mean"),
                    "comet_kiwi_mean": summary.get("comet_kiwi_mean"),
                    "xcomet_mean": summary.get("xcomet_mean"),
                }
            )

    index_path = run_root_base / f"{args.run_prefix}_index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    print(f"[temp-sweep] wrote {index_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
