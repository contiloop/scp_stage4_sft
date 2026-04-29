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

### 3. Configure access (HF / W&B / LLM API)

```sh
python -c "from huggingface_hub import login; login()"
python -m wandb login
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

### 5. Preprocess

```sh
make prepare-data CONFIG=configs/scp_stage4_real.yaml
```

### 6. Train

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
- Default profile remains mock-first for quick deterministic local checks.
- Real runtime requires additional dependencies/environment:
  - inference: `transformers`, `peft`, `torch`
  - training: `unsloth`, `trl`, `datasets`, `torch`
  - QE: `metricx24` or `comet` (or set `qe.primary.backend=heuristic`)
  - external API: `openai` package + `OPENAI_API_KEY` (for OpenAI provider)
