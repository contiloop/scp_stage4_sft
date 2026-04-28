SHELL := /bin/zsh
PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
PY := $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),$(PYTHON))
PYTHONPATH := src
CONFIG ?= configs/scp_stage4.yaml
RUN_ID ?= local_contract

.PHONY: set validate-config validate-jsonl validate-local test-local smoke-local \
	validate-remote-env smoke-remote-qe smoke-remote-model smoke-remote-api dry-run-remote-subset \
	prepare-data run-subset run-stage eval eval-ood \
	infer-q1 train-collapse-lora infer-q2 score call-api update-base

# Target: set
# required config keys: none
# input artifacts: none
# output artifacts: local directories for lightweight runs
# runtime: local CPU only, no GPU/API/QE required
# exit behavior: 0 on success; non-zero on directory/bootstrap failure
set:
	@mkdir -p artifacts/runs tests/fixtures src/scp_stage4
	@if [ ! -x "$(VENV_PYTHON)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	@if ! $(VENV_PYTHON) -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("pytest") else 1)'; then \
		$(VENV_PYTHON) -m pip install -q --upgrade pip pytest; \
	fi
	@$(VENV_PYTHON) -c 'import sys; print("set: python", sys.version.split()[0])'

# Target: validate-config
# required config keys: model.*, data.length.*, inference.q1/q2, pipeline.subset, training.backend, external_api.*, logging.local.*
# input artifacts: $(CONFIG)
# output artifacts: none
# runtime: local CPU only
# exit behavior: 0 if composed config contract is valid; non-zero on missing config/schema mismatch
validate-config:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.validate_config --config $(CONFIG)

# Target: validate-jsonl
# required config keys: logging.local.root_dir, run.run_id
# input artifacts: tests/fixtures/*.jsonl and/or artifacts/runs/$(RUN_ID)/**/*.jsonl
# output artifacts: none
# runtime: local CPU only (hooks optional Data/Schema validator when available)
# exit behavior: 0 if JSONL/schema contract passes; non-zero on malformed JSONL/schema mismatch
validate-jsonl:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.schema.validate_jsonl --config $(CONFIG) --run-id $(RUN_ID)

# Target: validate-local
# required config keys: same as validate-config + validate-jsonl requirements
# input artifacts: $(CONFIG), local fixtures/artifacts
# output artifacts: none
# runtime: local CPU only
# exit behavior: 0 if local contract validations pass; non-zero if any validation fails
validate-local: validate-config validate-jsonl

# Target: test-local
# required config keys: none (tests may compose config)
# input artifacts: tests/
# output artifacts: test reports in stdout
# runtime: local CPU only
# exit behavior: 0 if all tests pass; non-zero on any test failure
# note: pytest is bootstrapped by `make set`

test-local:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m pytest -q

# Target: smoke-local
# required config keys: run.run_id, logging.local.root_dir, pipeline.subset.*, qe.scoring.selection.default_rule.*, external_api.primary.*
# input artifacts: tests/fixtures/datapool.train.jsonl (optional; fallback fixture used when absent)
# output artifacts: artifacts/runs/$(RUN_ID)/subsets/subset_000/*.jsonl and run-level smoke summary
# runtime: local CPU only with mocked Q1/Q2/QE/API/update flow
# exit behavior: 0 on successful contract flow; non-zero on row-id drift/missing artifact/schema mismatch
smoke-local:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.smoke_local --config $(CONFIG) --run-id $(RUN_ID)

# Target: prepare-data
# required config keys: data.*, pipeline.subset.*, run.run_id
# input artifacts: tests/fixtures/*.jsonl (for local harness)
# output artifacts: artifacts/data/datapool.normalized.jsonl, datapool.train.jsonl, datapool.eval.jsonl, datapool.train.sampled.jsonl
# runtime: local CPU only, deterministic local normalization/split/sampling
# exit behavior: 0 on contract artifact generation; non-zero on config/schema/IO failures
prepare-data: validate-config
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.prepare_data --config $(CONFIG)

# Target: infer-q1
# required config keys: inference.q1.*, model.*, run.run_id
# input artifacts: subset input rows
# output artifacts: subsets/subset_000/q1.jsonl (mocked)
# runtime: local CPU only, mocked generation
# exit behavior: 0 on deterministic mocked output path readiness; non-zero on contract failure
infer-q1:
	@echo "infer-q1: mocked in smoke-local (no real model inference)"

# Target: train-collapse-lora
# required config keys: training.collapse_lora.*, training.backend
# input artifacts: subsets/subset_000/q1.jsonl
# output artifacts: subsets/subset_000/train_final/ (mocked marker)
# runtime: local CPU only, mocked training
# exit behavior: 0 on mocked contract pass; non-zero on contract failure
train-collapse-lora:
	@echo "train-collapse-lora: mocked in smoke-local (no real training)"

# Target: infer-q2
# required config keys: inference.q2.*, training.collapse_lora.*
# input artifacts: subsets/subset_000/q1.jsonl
# output artifacts: subsets/subset_000/q2.jsonl (mocked)
# runtime: local CPU only, mocked generation
# exit behavior: 0 on deterministic mocked output path readiness; non-zero on contract failure
infer-q2:
	@echo "infer-q2: mocked in smoke-local (no real collapse adapter inference)"

# Target: score
# required config keys: qe.*, pipeline.subset.*
# input artifacts: subsets/subset_000/q1.jsonl, q2.jsonl
# output artifacts: subsets/subset_000/scored.jsonl, selected.jsonl (mocked)
# runtime: local CPU only, mocked QE scoring
# exit behavior: 0 on deterministic mocked scoring pass; non-zero on contract failure
score:
	@echo "score: mocked in smoke-local (no real QE subprocess)"

# Target: call-api
# required config keys: external_api.*, logging.*
# input artifacts: subsets/subset_000/selected.jsonl
# output artifacts: subsets/subset_000/api_requests.jsonl, api.jsonl (mocked)
# runtime: local CPU only, mocked external API behavior
# exit behavior: 0 on deterministic mocked API contract pass; non-zero on contract failure
call-api:
	@echo "call-api: mocked in smoke-local (no real external API call)"

# Target: update-base
# required config keys: training.base_update.*, training.backend
# input artifacts: subsets/subset_000/api.jsonl
# output artifacts: subsets/subset_000/train_final/train_rows.jsonl (mocked)
# runtime: local CPU only, mocked training update
# exit behavior: 0 on deterministic mocked update pass; non-zero on contract failure
update-base:
	@echo "update-base: mocked in smoke-local (no real base model update)"

# Target: run-subset
# required config keys: full local harness config
# input artifacts: configs + local fixtures
# output artifacts: subset artifact chain under artifacts/runs/$(RUN_ID)
# runtime: local CPU only, mocked end-to-end subset flow
# exit behavior: 0 on full mocked subset contract pass; non-zero on any step contract failure
run-subset: prepare-data
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.smoke_local --config $(CONFIG) --run-id $(RUN_ID) --use-prepared-data

# Target: run-stage
# required config keys: full local harness config
# input artifacts: configs + local fixtures
# output artifacts: stage-level mocked outputs (single-subset foundation mode)
# runtime: local CPU only, mocked stage flow
# exit behavior: 0 on mocked stage contract pass; non-zero on any contract failure
run-stage: run-subset
	@echo "run-stage: foundation mode executes one mocked subset"

# Target: eval
# required config keys: pipeline.eval_after_subset.*, logging.*
# input artifacts: mocked subset outputs
# output artifacts: metrics JSONL entries (mocked)
# runtime: local CPU only
# exit behavior: 0 on mocked eval contract pass; non-zero on contract failure
eval:
	@echo "eval: mocked in foundation mode (no real metric runtime)"

# Target: eval-ood
# required config keys: pipeline.eval_after_subset.*, data.ood_test.*
# input artifacts: mocked subset outputs + ood config
# output artifacts: mocked OOD eval marker/logs
# runtime: local CPU only
# exit behavior: 0 on mocked OOD eval contract pass; non-zero on contract failure
eval-ood:
	@echo "eval-ood: mocked in foundation mode (no real reference-based evaluation)"

# Target: validate-remote-env
# required config keys: external_api.primary.api_key_env and full composed config validity
# input artifacts: $(CONFIG), environment variables
# output artifacts: none
# runtime: local/remote CPU only; no GPU/API call
# exit behavior: 0 when config/env contract parsing succeeds; non-zero on invalid config
validate-remote-env:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.remote_checks validate-env --config $(CONFIG)

# Target: smoke-remote-qe
# required config keys: qe.isolation.env.comet_python_env, qe.isolation.env.metricx_python_env
# input artifacts: $(CONFIG), QE env vars
# output artifacts: none
# runtime: remote preflight contract only; no real QE model execution
# exit behavior: 0 when required env vars/path contracts are present; non-zero otherwise
smoke-remote-qe:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.remote_checks smoke-qe --config $(CONFIG)

# Target: smoke-remote-model
# required config keys: training.backend
# input artifacts: $(CONFIG)
# output artifacts: none
# runtime: remote preflight contract only; no actual GPU model load
# exit behavior: 0 when model training contract is valid; non-zero otherwise
smoke-remote-model:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.remote_checks smoke-model --config $(CONFIG)

# Target: smoke-remote-api
# required config keys: external_api.primary.model, external_api.primary.api_key_env
# input artifacts: $(CONFIG), provider API key env var
# output artifacts: none
# runtime: remote preflight contract only; no real API request
# exit behavior: 0 when API contract is ready; non-zero on placeholder model/missing env
smoke-remote-api:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.remote_checks smoke-api --config $(CONFIG)

# Target: dry-run-remote-subset
# required config keys: same as smoke-local + remote contract validity
# input artifacts: $(CONFIG)
# output artifacts: artifacts/runs/dry_run_remote_subset/** mock subset artifacts
# runtime: remote deterministic dry-run using mocked flow only (no GPU/QE/API)
# exit behavior: 0 on successful dry-run artifact generation; non-zero on contract failure
dry-run-remote-subset:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.remote_checks dry-run-subset --config $(CONFIG)
