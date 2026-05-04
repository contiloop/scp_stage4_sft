# scp_stage4_sft

SCP (Self-Collapse Probing) Stage 4 — model-adaptive data construction loop for English→Korean translation. Probes the current model, expands only fragile samples via an external teacher, and updates the base model subset by subset.

This repository supports two runtime profiles:

- **`configs/scp_stage4.yaml` (default)**: contract-first mock runtime for fast local validation
- **`configs/scp_stage4_real.yaml`**: subprocess runtime for real inference/QE/API/training workers

## Quick start

### 1. Clone repository

```sh
git clone https://github.com/contiloop/scp_stage4_sft.git
cd scp_stage4_sft
```

### 2. Install dependencies

```sh
make set
make set-real-env
```

`make set` creates `.venv/`, installs `pytest`, and prepares local directories.
`make set-real-env` installs the real runtime stack (Unsloth / TRL / QE / API deps).
By default, it installs into the instance Python (`USE_VENV=0`). To install into `.venv`, use `make USE_VENV=1 set-real-env`.

### 3. Configure access (HF / W&B / LLM API)

```sh
python -c "from huggingface_hub import login; login()"
wandb login
export OPENAI_API_KEY="..."
```

If you use QE subprocess isolation, also set:

```sh
export COMET_PYTHON="/path/to/comet-env/bin/python"
export METRICX_PYTHON="/path/to/metricx-env/bin/python"
```

### 4. Validate setup

```sh
make validate-local
make validate-real-config
make test-local
```

### 4A. Optional: tokenizer CPU parallelism for `prepare-data`

For large runs with `data.length.mode=tokenizer`, enabling tokenizer parallelism often speeds up normalization.

```sh
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=16   # set to your CPU core count (or slightly lower)
```

Notes:

- Usually helpful when running one large `prepare-data` job.
- Not always better if many CPU-heavy jobs run concurrently (oversubscription can hurt throughput).
- If memory pressure is high, reduce `RAYON_NUM_THREADS`.

### 5. Preprocess

```sh
make prepare-data CONFIG=configs/scp_stage4_real.yaml
```

### 5A. Publish processed data bundle to HF dataset repo (recommended)

This packages and uploads:

- `datapool.normalized.parquet`
- `datapool.train.parquet`
- `datapool.eval.parquet`
- `prepare_data_summary.json`
- `effective_config.yaml`
- `config_hash.txt`
- `prepared_manifest.json`

Optional compatibility artifacts during migration:

- `datapool.normalized.jsonl`
- `datapool.train.jsonl`
- `datapool.eval.jsonl`
- `datapool.train.sampled.parquet`
- `datapool.train.sampled.jsonl`

```sh
DATASET_REPO="<org_or_user>/scp-stage4-prepared"
BUNDLE_TAG="v2026-04-30-real"

make pack-prepared-data \
  CONFIG=configs/scp_stage4_real.yaml \
  PREPARED_BUNDLE_TAG="${BUNDLE_TAG}"

make upload-prepared-data \
  HF_DATASET_REPO="${DATASET_REPO}" \
  PREPARED_BUNDLE_TAG="${BUNDLE_TAG}" \
  HF_DATASET_PATH="prepared/${BUNDLE_TAG}" \
  HF_DATASET_REVISION=main \
  HF_DATASET_TAG="${BUNDLE_TAG}"
```

`HF_DATASET_TAG` creates a Hub git tag on the uploaded commit so the bundle can be pinned by immutable revision later.
If you intentionally reuse an existing tag name, add `HF_DATASET_TAG_EXIST_OK=1`.

### 5B. Reuse processed data on a new instance (skip prepare-data)

```sh
DATASET_REPO="<org_or_user>/scp-stage4-prepared"
BUNDLE_TAG="v2026-04-30-real"

make download-prepared-data \
  HF_DATASET_REPO="${DATASET_REPO}" \
  HF_DATASET_PATH="prepared/${BUNDLE_TAG}" \
  HF_DATASET_REVISION="${BUNDLE_TAG}"
```

This restore path materializes parquet datapool artifacts (`train/eval/normalized`) for execution.

Then start directly from subset/stage execution (parquet-first):

```sh
make run-subset-real-from-prepared RUN_ID=real_subset_001
make run-stage-real-from-prepared RUN_ID=real_stage_001
```

Check source mix ratio anytime:

```sh
make data-source-ratio
```

### 6. Train (standard path with local preprocess)

Run one subset:

```sh
make run-subset-real RUN_ID=real_subset_001
```

Run a full stage:

```sh
make run-stage-real RUN_ID=real_stage_001
```

The real pipeline follows:

```txt
prepare-data → infer-q1 → train-collapse-lora → infer-q2 → score
             → unload-collapse-lora → call-api → update-base
```

### 7. Upload artifacts/checkpoints (optional)

Checkpoint and run artifacts are written under:

```sh
artifacts/runs/<RUN_ID>/
```

Upload to your model hub / storage policy as needed. (No mandatory upload target is enforced by this repo.)

### 8. Mock profile (optional)

For fast local contract checks without GPU/API:

```sh
make smoke-local
```

## Running a single step

Each DAG node is its own Make target with explicit dependencies, so you can stop at any point:

```sh
make infer-q1                  # prepare-data → infer-q1
make score                     # …→ infer-q2 → score
make run-subset                # full subset chain on subset_000
make run-stage                 # all subsets in the configured stage
```

Override the run id or config:

```sh
make smoke-local RUN_ID=my_run CONFIG=configs/scp_stage4.yaml
```

Pass extra CLI flags through `OVERRIDES`:

```sh
make smoke-local OVERRIDES="--set pipeline.subset.size=8"
```

## Layout

| Path | Purpose |
| --- | --- |
| `src/scp_stage4/pipeline/` | DAG entrypoints (`prepare_data`, `step_subset`, `smoke_local`, `remote_checks`) |
| `src/scp_stage4/config/` | Config loader + fail-fast validator |
| `src/scp_stage4/schema/` | JSONL/logging schema validators |
| `configs/` | Composed pipeline configs (`scp_stage4.yaml`, `pipeline.yaml`, …) |
| `tests/fixtures/` | Datapool fixtures used by smoke and unit tests |
| `artifacts/runs/<run_id>/` | Per-run subset artifacts (`q1.jsonl`, `q2.jsonl`, `scored.jsonl`, `selected.jsonl`, `api.jsonl`, `train_final/`) |
| `docs/` | Design docs (overview, data pipeline, config schema, QE, logging, …) |

## Docs

- [docs/scp-overview.md](docs/scp-overview.md) — motivation, SCP loop, research scope
- [docs/data-pipeline.md](docs/data-pipeline.md) — subset / collapse / update model
- [docs/config-schema.md](docs/config-schema.md) — composed config contract
- [docs/qe-scoring.md](docs/qe-scoring.md), [docs/qe-isolation.md](docs/qe-isolation.md) — QE selection rules and runtime isolation
- [docs/logging.md](docs/logging.md) — structured event contract
- [AGENTS.md](AGENTS.md) — operator-facing contract for agents driving the pipeline

## Notes

- The package is not pip-installed; the Makefile sets `PYTHONPATH=src` for every target. If you invoke `pytest` directly, prefix with `PYTHONPATH=src`.
- Default Make behavior uses the instance Python (`USE_VENV=0`). To force `.venv`, pass `USE_VENV=1` (example: `make USE_VENV=1 validate-real-config`).
- Default profile remains mock-first for quick deterministic local checks.
- Real runtime requires additional dependencies/environment:
  - inference: `transformers`, `peft`, `torch`
  - training: `unsloth`, `trl`, `datasets`, `torch`
  - QE: `metricx24` or `comet` (or set `qe.primary.backend=heuristic`)
  - external API: `openai` package + `OPENAI_API_KEY` (for OpenAI provider)
