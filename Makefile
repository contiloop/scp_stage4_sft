SHELL := /bin/sh
PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
USE_VENV ?= 0
PY := $(if $(filter 1,$(USE_VENV)),$(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),$(PYTHON)),$(PYTHON))
REAL_ENV_PY := $(if $(filter 1,$(USE_VENV)),$(VENV_PYTHON),$(PYTHON))
SETUP_PY := $(if $(filter 1,$(USE_VENV)),$(VENV_PYTHON),$(PYTHON))
QE_VENV_DIR ?= $(HOME)/.venvs/comet
PYTHONPATH := src
CONFIG ?= configs/scp_stage4.yaml
RUN_ID ?= local_contract
OVERRIDES ?=
PREPARED_BUNDLE_ROOT ?= artifacts/prepared_data_bundles
PREPARED_BUNDLE_TAG ?=
PREPARED_BUNDLE_DIR ?=
HF_DATASET_REPO ?=
HF_DATASET_PATH ?=
HF_DATASET_REVISION ?= main
HF_DATASET_TAG ?=
HF_DATASET_TAG_MESSAGE ?=
HF_DATASET_TAG_EXIST_OK ?= 0
HF_DATASET_PRIVATE ?= 0
HF_CREATE_REPO ?= 1
HF_COMMIT_MESSAGE ?=
HF_CHECKPOINT_REPO ?= alwaysgood/scp-stage4-run-main-001
HF_CHECKPOINT_REPO_TYPE ?= dataset
HF_CHECKPOINT_REVISION ?= main
REEVAL_CONFIG ?= configs/scp_stage4_real_1gpu_greedy_eval.yaml
REEVAL_RUN_ID ?= greedy_reeval_main_001
REEVAL_CHECKPOINT_INDICES ?= 17 19 31 32
REPLAY_REPO ?= alwaysgood/scp-stage4-run-main-001
REPLAY_REPO_TYPE ?= dataset
REPLAY_REVISION ?= main
REPLAY_CONFIG ?= configs/scp_stage4_real_1gpu_greedy_eval.yaml
REPLAY_RUN_ID ?= replay_main_001_greedy
REPLAY_START_SUBSET ?= 0
REPLAY_END_SUBSET ?= 32
REPLAY_EXTRA_ARGS ?=
TRANSLATE_CHECKPOINT ?=
TRANSLATE_TEXT ?=
TRANSLATE_INPUT_FILE ?=
TRANSLATE_OUTPUT ?=
SKIP_CAUSAL_CONV1D ?= 0
TORCH_INDEX_URL ?= https://download.pytorch.org/whl/cu128
PIN_TORCH_VERSION ?= 2.10.0
PIN_TORCHVISION_VERSION ?= 0.25.0
PIN_TORCHAUDIO_VERSION ?= 2.10.0
PIN_TRANSFORMERS_VERSION ?= 5.5.0
PIN_TRL_VERSION ?= 0.24.0
PIN_DATASETS_VERSION ?= 3.4.1
PIN_UNSLOTH_VERSION ?= 2026.5.2
PIN_UNSLOTH_ZOO_VERSION ?= 2026.5.1
PIN_VLLM_VERSION ?= 0.19.1
PIN_HF_HUB_VERSION ?= 1.14.0
PIN_HF_XET_VERSION ?= 1.4.3
PIN_SETUPTOOLS_SPEC ?= "setuptools>=77.0.3,<81.0.0"

# FlashAttention2 wheel hosted in a HF dataset. Built once for torch 2.10/cu128/py310/sm_80
# (A100). Override FLASH_ATTN_REPO/FILE to use a different host or rebuild for new arches.
FLASH_ATTN_REPO ?= alwaysgood/scp-stage4-wheels
FLASH_ATTN_WHL ?= flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl

.PHONY: set set-real-env validate-config validate-jsonl validate-local test-local smoke-local \
	validate-remote-env smoke-remote-qe smoke-remote-model smoke-remote-api dry-run-remote-subset \
	validate-real-config run-subset-real run-stage-real run-subset-real-from-prepared run-stage-real-from-prepared \
	prepare-data run-subset run-stage eval eval-ood reeval-greedy-checkpoints replay-saved-updates translate-checkpoint data-source-ratio \
	infer-q1 train-collapse-lora infer-q2 score unload-collapse-lora call-api update-base \
	pack-prepared-data upload-prepared-data download-prepared-data verify-cuda-kernels

# Target: set
# required config keys: none
# input artifacts: none
# output artifacts: local directories for lightweight runs
# runtime: local CPU only, no GPU/API/QE required
# exit behavior: 0 on success; non-zero on directory/bootstrap failure
set:
	@mkdir -p artifacts/runs tests/fixtures src/scp_stage4
	@if [ "$(USE_VENV)" = "1" ] && [ ! -x "$(VENV_PYTHON)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	@if ! $(SETUP_PY) -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("pytest") else 1)'; then \
		$(SETUP_PY) -m pip install --upgrade pip pytest; \
	fi
	@$(SETUP_PY) -c 'import sys; print("set:", sys.executable, sys.version.split()[0])'

# Target: set-real-env
# required config keys: none
# input artifacts: none
# output artifacts: selected python environment with runtime deps for real subprocess workers
# runtime: local/remote machine setup step; downloads packages and may require CUDA-compatible wheels
# exit behavior: 0 on successful dependency install; non-zero on package resolver/install failure
set-real-env:
	@if [ "$(USE_VENV)" = "1" ] && [ ! -x "$(VENV_PYTHON)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	@$(REAL_ENV_PY) -m pip install --upgrade pip
	@$(REAL_ENV_PY) -m pip install $(PIN_SETUPTOOLS_SPEC)
	@$(REAL_ENV_PY) -m pip install \
		--index-url $(TORCH_INDEX_URL) \
		"torch==$(PIN_TORCH_VERSION)" \
		"torchvision==$(PIN_TORCHVISION_VERSION)" \
		"torchaudio==$(PIN_TORCHAUDIO_VERSION)"
	@$(REAL_ENV_PY) -m pip install \
		"trl==$(PIN_TRL_VERSION)" \
		"datasets==$(PIN_DATASETS_VERSION)"
	@$(REAL_ENV_PY) -m pip install \
		"unsloth-zoo==$(PIN_UNSLOTH_ZOO_VERSION)" \
		"unsloth==$(PIN_UNSLOTH_VERSION)"
	@$(REAL_ENV_PY) -m pip uninstall -y vllm || true
	@$(REAL_ENV_PY) -m pip install \
		"vllm==$(PIN_VLLM_VERSION)" \
		--extra-index-url $(TORCH_INDEX_URL)
	@$(REAL_ENV_PY) -m pip install --index-url $(TORCH_INDEX_URL) "xformers==0.0.34"
	@$(REAL_ENV_PY) -m pip install \
		tokenizers hydra-core omegaconf \
		openai peft wandb sacrebleu \
		sentencepiece bitsandbytes hf_transfer msgspec tyro torchao
	# vLLM 0.19.1 requires transformers.tokenization_utils_tokenizers.TokenizersBackend,
	# which exists in transformers 5.x. Force-install transformers/hf_hub at the end with
	# --no-deps so upstream pins from trl/unsloth/vllm don't roll us back to 4.x.
	@$(REAL_ENV_PY) -m pip install --no-deps \
		"transformers==$(PIN_TRANSFORMERS_VERSION)" \
		"huggingface_hub>=$(PIN_HF_HUB_VERSION),<2" \
		"hf-xet>=$(PIN_HF_XET_VERSION),<2"
	# FlashAttention2: no torch 2.10 prebuilt on Dao-AILab; install our self-built wheel
	# hosted on HF datasets (sm_80, A100). Override FLASH_ATTN_* to swap wheel.
	@$(REAL_ENV_PY) -m pip install \
		"https://huggingface.co/datasets/$(FLASH_ATTN_REPO)/resolve/main/$(FLASH_ATTN_WHL)"
	@if $(REAL_ENV_PY) -m pip install weave; then \
		echo "  weave_install_ok=true"; \
	else \
		echo "  weave_install_ok=false"; \
	fi
	@if [ "$(SKIP_CAUSAL_CONV1D)" = "1" ]; then \
		echo "  skip causal_conv1d setup (SKIP_CAUSAL_CONV1D=1)"; \
	else \
		PYTHON=$(REAL_ENV_PY) bash scripts/ensure_causal_conv1d.sh; \
	fi
	@$(REAL_ENV_PY) -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule" 2>/dev/null \
		|| $(REAL_ENV_PY) -m pip install flash-linear-attention
	@$(MAKE) verify-cuda-kernels REAL_ENV_PY=$(REAL_ENV_PY) SKIP_CAUSAL_CONV1D=$(SKIP_CAUSAL_CONV1D)
	@$(REAL_ENV_PY) -c 'import sys, torch; print("set-real-env:", sys.executable, "torch", torch.__version__)'
	@echo "set-real-env: setting up QE isolation venv at $(QE_VENV_DIR)..."
	@if [ ! -x "$(QE_VENV_DIR)/bin/python" ]; then \
		$(PYTHON) -m venv --without-pip $(QE_VENV_DIR) && \
		curl -sS https://bootstrap.pypa.io/get-pip.py | $(QE_VENV_DIR)/bin/python; \
	fi
	@$(QE_VENV_DIR)/bin/python -m pip install --upgrade pip setuptools wheel
	@$(QE_VENV_DIR)/bin/pip install \
		torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
	@$(QE_VENV_DIR)/bin/pip install --no-deps transformers
	@$(QE_VENV_DIR)/bin/pip install \
		sentencepiece safetensors accelerate huggingface_hub \
		"unbabel-comet>=2.2.7" sacrebleu
	@$(QE_VENV_DIR)/bin/python -c 'import torch; print("set-real-env: QE venv torch", torch.__version__, "cuda", torch.cuda.is_available())'
	@echo "set-real-env: export COMET_PYTHON=$(QE_VENV_DIR)/bin/python"
	@echo "set-real-env: export METRICX_PYTHON=$(QE_VENV_DIR)/bin/python"

# Target: verify-cuda-kernels
# required config keys: none
# input artifacts: installed runtime python packages in REAL_ENV_PY
# output artifacts: stdout kernel readiness status
# runtime: local/remote GPU runtime check
# exit behavior: 0 if kernel checks pass (or skip flag enabled); non-zero on kernel check failure
verify-cuda-kernels:
	@if [ "$(SKIP_CAUSAL_CONV1D)" = "1" ]; then \
		echo "  skip CUDA kernel verification (SKIP_CAUSAL_CONV1D=1)"; \
	else \
		PYTHON=$(REAL_ENV_PY) bash scripts/verify_cuda_kernels.sh; \
	fi

# Target: validate-config
# required config keys: model.*, data.length.*, inference.q1/q2, pipeline.subset, training.backend, external_api.*, logging.local.*
# input artifacts: $(CONFIG)
# output artifacts: none
# runtime: local CPU only
# exit behavior: 0 if composed config contract is valid; non-zero on missing config/schema mismatch
validate-config:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.validate_config --config $(CONFIG) $(OVERRIDES)

# Target: validate-jsonl
# required config keys: logging.local.root_dir, run.run_id
# input artifacts: tests/fixtures/*.jsonl and/or artifacts/runs/$(RUN_ID)/**/*.jsonl
# output artifacts: none
# runtime: local CPU only (hooks optional Data/Schema validator when available)
# exit behavior: 0 if JSONL/schema contract passes; non-zero on malformed JSONL/schema mismatch
validate-jsonl:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.schema.validate_jsonl --config $(CONFIG) --run-id $(RUN_ID) $(OVERRIDES)

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
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.smoke_local --config $(CONFIG) --run-id $(RUN_ID) $(OVERRIDES)

# Target: prepare-data
# required config keys: data.*, pipeline.subset.*, run.run_id
# input artifacts: tests/fixtures/*.jsonl (for local harness)
# output artifacts: artifacts/data/datapool.normalized.jsonl, datapool.train.jsonl, datapool.eval.jsonl, datapool.train.sampled.jsonl
# runtime: local CPU only, deterministic local normalization/split/sampling
# exit behavior: 0 on contract artifact generation; non-zero on config/schema/IO failures
prepare-data: validate-config
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.prepare_data --config $(CONFIG) $(OVERRIDES)

# Target: infer-q1
# required config keys: inference.q1.*, model.*, run.run_id
# input artifacts: subset input rows
# output artifacts: subsets/subset_000/q1.jsonl (mocked)
# runtime: local CPU only, mocked generation
# exit behavior: 0 on deterministic mocked output path readiness; non-zero on contract failure
infer-q1:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.step_subset infer-q1 --config $(CONFIG) --run-id $(RUN_ID) --subset-idx 0 --use-prepared-data $(OVERRIDES)

# Target: train-collapse-lora
# required config keys: training.collapse_lora.*, training.backend
# input artifacts: subsets/subset_000/q1.jsonl
# output artifacts: subsets/subset_000/collapse_adapter/collapse_state.json
# runtime: local CPU only in mock mode; real mode delegates to training subprocess
# exit behavior: 0 on collapse adapter state; non-zero on contract failure
train-collapse-lora: infer-q1
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.step_subset train-collapse-lora --config $(CONFIG) --run-id $(RUN_ID) --subset-idx 0 $(OVERRIDES)

# Target: infer-q2
# required config keys: inference.q2.*, training.collapse_lora.*
# input artifacts: subsets/subset_000/q1.jsonl
# output artifacts: subsets/subset_000/q2.jsonl (mocked)
# runtime: local CPU only, mocked generation
# exit behavior: 0 on deterministic mocked output path readiness; non-zero on contract failure
infer-q2: train-collapse-lora
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.step_subset infer-q2 --config $(CONFIG) --run-id $(RUN_ID) --subset-idx 0 $(OVERRIDES)

# Target: score
# required config keys: qe.*, pipeline.subset.*
# input artifacts: subsets/subset_000/q1.jsonl, q2.jsonl
# output artifacts: subsets/subset_000/scored.jsonl, selected.jsonl (mocked)
# runtime: local CPU only, mocked QE scoring
# exit behavior: 0 on deterministic mocked scoring pass; non-zero on contract failure
score: infer-q2
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.step_subset score --config $(CONFIG) --run-id $(RUN_ID) --subset-idx 0 $(OVERRIDES)

# Target: unload-collapse-lora
# required config keys: training.runtime.*, training.collapse_lora.*
# input artifacts: subsets/subset_000/q2.jsonl, collapse adapter state
# output artifacts: subsets/subset_000/clean_base.json
# runtime: local CPU only in mock mode; real mode delegates to training subprocess
# exit behavior: 0 after clean-base verification; non-zero on contract failure
unload-collapse-lora: score
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.step_subset unload-collapse-lora --config $(CONFIG) --run-id $(RUN_ID) --subset-idx 0 $(OVERRIDES)

# Target: call-api
# required config keys: external_api.*, logging.*
# input artifacts: subsets/subset_000/selected.jsonl
# output artifacts: subsets/subset_000/api_requests.jsonl, api.jsonl (mocked)
# runtime: local CPU only, mocked external API behavior
# exit behavior: 0 on deterministic mocked API contract pass; non-zero on contract failure
call-api: unload-collapse-lora
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.step_subset call-api --config $(CONFIG) --run-id $(RUN_ID) --subset-idx 0 $(OVERRIDES)

# Target: update-base
# required config keys: training.base_update.*, training.backend
# input artifacts: subsets/subset_000/api.jsonl
# output artifacts: subsets/subset_000/train_final/train_rows.jsonl (mocked)
# runtime: local CPU only, mocked training update
# exit behavior: 0 on deterministic mocked update pass; non-zero on contract failure
update-base: call-api
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.step_subset update-base --config $(CONFIG) --run-id $(RUN_ID) --subset-idx 0 $(OVERRIDES)

# Target: run-subset
# required config keys: full local harness config
# input artifacts: configs + local fixtures
# output artifacts: subset artifact chain under artifacts/runs/$(RUN_ID)
# runtime: local CPU only, mocked end-to-end subset flow
# exit behavior: 0 on full mocked subset contract pass; non-zero on any step contract failure
run-subset: prepare-data
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.step_subset run-subset --config $(CONFIG) --run-id $(RUN_ID) --subset-idx 0 --use-prepared-data $(OVERRIDES)

# Target: run-stage
# required config keys: full local harness config
# input artifacts: configs + local fixtures
# output artifacts: stage-level subset chain and run_stage_summary.json
# runtime: local CPU only in mock mode; real backends use configured subprocess hooks
# exit behavior: 0 when every scheduled subset completes; non-zero on any contract failure
run-stage: prepare-data
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.step_subset run-stage --config $(CONFIG) --run-id $(RUN_ID) $(OVERRIDES)

# Target: eval
# required config keys: pipeline.eval_after_subset.*, logging.*
# input artifacts: subset update-base checkpoint + artifacts/data/ood_test.jsonl
# output artifacts: artifacts/runs/$(RUN_ID)/eval/ood_test/subset_000.{rows,summary}.jsonl/json
# runtime: inference + QE backends according to config runtime modes
# exit behavior: 0 on successful OOD eval; non-zero on inference/QE contract failure
eval: prepare-data
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.step_subset eval-ood --config $(CONFIG) --run-id $(RUN_ID) --subset-idx 0 $(OVERRIDES)

# Target: eval-ood
# required config keys: pipeline.eval_after_subset.*, data.ood_test.*
# input artifacts: subset update-base checkpoint + OOD reference set
# output artifacts: artifacts/runs/$(RUN_ID)/eval/ood_test/subset_000.{rows,summary}.jsonl/json
# runtime: inference + QE subprocess/mock backends per config
# exit behavior: 0 on successful reference-based eval; non-zero on runtime/contract failure
eval-ood: eval

# Target: reeval-greedy-checkpoints
# required config keys: inference.eval.*, pipeline.eval_after_subset.*, qe.*
# input artifacts: HF checkpoint subset archives (default alwaysgood/scp-stage4-run-main-001, subsets 17/19/31/32)
# output artifacts: artifacts/runs/$(RUN_ID)/eval/ood_test/subset_*.{rows,summary}.jsonl/json
# runtime: remote GPU inference + QE subprocess; network required for HF archive download
# exit behavior: 0 after every requested checkpoint eval succeeds; non-zero on download/extract/eval failure
reeval-greedy-checkpoints:
	@PYTHONPATH=$(PYTHONPATH) $(PY) scripts/reeval_greedy_checkpoints.py \
		--repo-id $(HF_CHECKPOINT_REPO) \
		--repo-type $(HF_CHECKPOINT_REPO_TYPE) \
		--revision $(HF_CHECKPOINT_REVISION) \
		--config $(REEVAL_CONFIG) \
		--run-id $(REEVAL_RUN_ID) \
		--checkpoint-indices $(REEVAL_CHECKPOINT_INDICES) \
		$(OVERRIDES)

# Target: replay-saved-updates
# required config keys: training.base_update.*, training.runtime.subprocess.update_command, inference.eval.*, pipeline.eval_after_subset.*
# input artifacts: saved subset api.jsonl + clean_base.json from HF run repo
# output artifacts: artifacts/runs/$(REPLAY_RUN_ID)/subsets/subset_*/train_final/checkpoint_state.json and eval/ood_test/*.json/jsonl
# runtime: remote GPU training + greedy inference + QE subprocess; network required for HF restore unless --skip-download
# exit behavior: 0 after every requested subset update+eval succeeds; non-zero on restore/train/eval failure
replay-saved-updates:
	@PYTHONPATH=$(PYTHONPATH) $(PY) scripts/replay_saved_updates.py \
		--repo-id $(REPLAY_REPO) \
		--repo-type $(REPLAY_REPO_TYPE) \
		--revision $(REPLAY_REVISION) \
		--config $(REPLAY_CONFIG) \
		--run-id $(REPLAY_RUN_ID) \
		--start-subset $(REPLAY_START_SUBSET) \
		--end-subset $(REPLAY_END_SUBSET) \
		$(REPLAY_EXTRA_ARGS) \
		$(OVERRIDES)

# Target: translate-checkpoint
# required config keys: model.*, inference.eval.*, prompts.*
# input artifacts: local checkpoint path in TRANSLATE_CHECKPOINT and text via TRANSLATE_TEXT or TRANSLATE_INPUT_FILE
# output artifacts: optional TRANSLATE_OUTPUT JSONL; translation is printed to stdout
# runtime: remote GPU inference through vLLM subprocess
# exit behavior: 0 after one translation succeeds; non-zero on missing checkpoint or inference failure
translate-checkpoint:
	@test -n "$(TRANSLATE_CHECKPOINT)" || (echo "TRANSLATE_CHECKPOINT is required" >&2; exit 2)
	@PYTHONPATH=$(PYTHONPATH) $(PY) scripts/translate_with_checkpoint.py \
		--checkpoint "$(TRANSLATE_CHECKPOINT)" \
		--config "$(REEVAL_CONFIG)" \
		$(if $(strip $(TRANSLATE_TEXT)),--text "$(TRANSLATE_TEXT)",) \
		$(if $(strip $(TRANSLATE_INPUT_FILE)),--input-file "$(TRANSLATE_INPUT_FILE)",) \
		$(if $(strip $(TRANSLATE_OUTPUT)),--output "$(TRANSLATE_OUTPUT)",) \
		$(OVERRIDES)

# Target: data-source-ratio
# required config keys: none
# input artifacts: artifacts/data/datapool.{train,eval,normalized}.jsonl (existing files only)
# output artifacts: none (prints per-dataset ratio report)
# runtime: local CPU only
# exit behavior: 0 when at least one requested file is reported; non-zero when none exist
data-source-ratio:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.data_source_ratio

# Target: validate-remote-env
# required config keys: external_api.primary.api_key_env and full composed config validity
# input artifacts: $(CONFIG), environment variables
# output artifacts: none
# runtime: local/remote CPU only; no GPU/API call
# exit behavior: 0 when config/env contract parsing succeeds; non-zero on invalid config
validate-remote-env:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.remote_checks validate-env --config $(CONFIG) $(OVERRIDES)

# Target: smoke-remote-qe
# required config keys: qe.isolation.env.comet_python_env, qe.isolation.env.metricx_python_env
# input artifacts: $(CONFIG), QE env vars
# output artifacts: none
# runtime: remote preflight contract only; no real QE model execution
# exit behavior: 0 when required env vars/path contracts are present; non-zero otherwise
smoke-remote-qe:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.remote_checks smoke-qe --config $(CONFIG) $(OVERRIDES)

# Target: smoke-remote-model
# required config keys: training.backend
# input artifacts: $(CONFIG)
# output artifacts: none
# runtime: remote preflight contract only; no actual GPU model load
# exit behavior: 0 when model training contract is valid; non-zero otherwise
smoke-remote-model:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.remote_checks smoke-model --config $(CONFIG) $(OVERRIDES)

# Target: smoke-remote-api
# required config keys: external_api.primary.model, external_api.primary.api_key_env
# input artifacts: $(CONFIG), provider API key env var
# output artifacts: none
# runtime: remote preflight contract only; no real API request
# exit behavior: 0 when API contract is ready; non-zero on placeholder model/missing env
smoke-remote-api:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.remote_checks smoke-api --config $(CONFIG) $(OVERRIDES)

# Target: dry-run-remote-subset
# required config keys: same as smoke-local + remote contract validity
# input artifacts: $(CONFIG)
# output artifacts: artifacts/runs/dry_run_remote_subset/** mock subset artifacts
# runtime: remote deterministic dry-run using mocked flow only (no GPU/QE/API)
# exit behavior: 0 on successful dry-run artifact generation; non-zero on contract failure
dry-run-remote-subset:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.remote_checks dry-run-subset --config $(CONFIG) $(OVERRIDES)

# Target: validate-real-config
# required config keys: full subprocess runtime commands across inference/qe/external_api/training
# input artifacts: configs/scp_stage4_real.yaml
# output artifacts: none
# runtime: local CPU only (contract validation)
# exit behavior: 0 when real profile config is structurally valid; non-zero otherwise
validate-real-config:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.validate_config --config configs/scp_stage4_real.yaml $(OVERRIDES)

# Target: run-subset-real
# required config keys: same as run-subset + subprocess worker commands
# input artifacts: prepared datapool + runtime deps in active python env
# output artifacts: full subset artifact chain under artifacts/runs/$(RUN_ID)
# runtime: subprocess backends for inference/QE/API/training
# exit behavior: 0 on successful subset completion; non-zero with structured failure logs
run-subset-real: prepare-data
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.step_subset run-subset --config configs/scp_stage4_real.yaml --run-id $(RUN_ID) --subset-idx 0 --use-prepared-data $(OVERRIDES)

# Target: run-stage-real
# required config keys: same as run-stage + subprocess worker commands
# input artifacts: prepared datapool + runtime deps in active python env
# output artifacts: run_stage_summary.json + per-subset artifacts
# runtime: subprocess backends for inference/QE/API/training
# exit behavior: 0 when all subsets complete; non-zero on first contract/runtime failure
run-stage-real: prepare-data
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.step_subset run-stage --config configs/scp_stage4_real.yaml --run-id $(RUN_ID) $(OVERRIDES)

# Target: run-subset-real-from-prepared
# required config keys: same as run-subset-real
# input artifacts: artifacts/data/datapool.train*.parquet restored from prepared bundle (jsonl fallback allowed)
# output artifacts: full subset artifact chain under artifacts/runs/$(RUN_ID)
# runtime: subprocess backends for inference/QE/API/training
# exit behavior: 0 on successful subset completion; non-zero with structured failure logs
run-subset-real-from-prepared:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.step_subset run-subset --config configs/scp_stage4_real.yaml --run-id $(RUN_ID) --subset-idx 0 --use-prepared-data $(OVERRIDES)

# Target: run-stage-real-from-prepared
# required config keys: same as run-stage-real
# input artifacts: artifacts/data/datapool.train*.parquet restored from prepared bundle (jsonl fallback allowed)
# output artifacts: run_stage_summary.json + per-subset artifacts
# runtime: subprocess backends for inference/QE/API/training
# exit behavior: 0 when all subsets complete; non-zero on first contract/runtime failure
run-stage-real-from-prepared:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.step_subset run-stage --config configs/scp_stage4_real.yaml --run-id $(RUN_ID) $(OVERRIDES)

# Target: pack-prepared-data
# required config keys: full data preparation config used for this datapool
# input artifacts: artifacts/data/{datapool.normalized.parquet,datapool.train.parquet,datapool.eval.parquet,prepare_data_summary.json}
# output artifacts: artifacts/prepared_data_bundles/<tag>/ + manifest/effective_config/config_hash
# runtime: local CPU only
# exit behavior: 0 on successful bundle creation; non-zero on missing artifacts/config mismatch
pack-prepared-data:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.prepared_data_hub pack \
		--config $(CONFIG) \
		--artifacts-dir artifacts/data \
		--output-root $(PREPARED_BUNDLE_ROOT) \
		$(if $(strip $(PREPARED_BUNDLE_TAG)),--tag $(PREPARED_BUNDLE_TAG),) \
		$(OVERRIDES)

# Target: upload-prepared-data
# required config keys: none (uses packaged bundle + HF auth)
# input artifacts: one bundle directory under artifacts/prepared_data_bundles/
# output artifacts: uploaded bundle at HF dataset repo path (optionally tagged)
# runtime: network + HF token required
# exit behavior: 0 on successful upload; non-zero on missing repo/bundle/auth failures
upload-prepared-data:
	@test -n "$(HF_DATASET_REPO)" || (echo "HF_DATASET_REPO is required" >&2; exit 2)
	@bundle_dir="$(PREPARED_BUNDLE_DIR)"; \
	if [ -z "$$bundle_dir" ]; then \
		test -n "$(PREPARED_BUNDLE_TAG)" || (echo "Set PREPARED_BUNDLE_DIR or PREPARED_BUNDLE_TAG" >&2; exit 2); \
		bundle_dir="$(PREPARED_BUNDLE_ROOT)/$(PREPARED_BUNDLE_TAG)"; \
	fi; \
	PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.prepared_data_hub upload \
		--repo-id "$(HF_DATASET_REPO)" \
		--bundle-dir "$$bundle_dir" \
		--revision "$(HF_DATASET_REVISION)" \
		$(if $(strip $(HF_DATASET_PATH)),--path-in-repo $(HF_DATASET_PATH),) \
		$(if $(filter 1 true TRUE yes YES,$(HF_DATASET_PRIVATE)),--private,) \
		$(if $(filter 0 false FALSE no NO,$(HF_CREATE_REPO)),--no-create-repo,) \
		$(if $(strip $(HF_COMMIT_MESSAGE)),--commit-message "$(HF_COMMIT_MESSAGE)",) \
		$(if $(strip $(HF_DATASET_TAG)),--tag $(HF_DATASET_TAG),) \
		$(if $(strip $(HF_DATASET_TAG_MESSAGE)),--tag-message "$(HF_DATASET_TAG_MESSAGE)",) \
		$(if $(filter 1 true TRUE yes YES,$(HF_DATASET_TAG_EXIST_OK)),--tag-exist-ok,)

# Target: download-prepared-data
# required config keys: none (uses HF dataset path/revision)
# input artifacts: HF dataset bundle path containing required prepared artifacts
# output artifacts: artifacts/data/*.parquet (+ optional *.jsonl) + prepare_data_summary.json + effective_config/config_hash/manifest
# runtime: network + HF token for private repos
# exit behavior: 0 on successful restore; non-zero on repo/path/revision mismatches
download-prepared-data:
	@test -n "$(HF_DATASET_REPO)" || (echo "HF_DATASET_REPO is required" >&2; exit 2)
	@test -n "$(HF_DATASET_PATH)" || (echo "HF_DATASET_PATH is required (ex: prepared/v2026-04-30)" >&2; exit 2)
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m scp_stage4.pipeline.prepared_data_hub download \
		--repo-id "$(HF_DATASET_REPO)" \
		--path-in-repo "$(HF_DATASET_PATH)" \
		--revision "$(HF_DATASET_REVISION)" \
		--output-dir artifacts/data \
		--local-download-dir artifacts/prepared_data_download
