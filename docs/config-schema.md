# SCP Stage 4 Config Schema

> **Project**: `scp_stage4_sft`  
> **Scope**: YAML configuration ownership, split config files, effective config merging, validation rules, and schema-level invariants.

---

## 1. Goal

This document defines the configuration contract for SCP Stage 4.

All pipeline behavior must be controlled through YAML config. Code should read a merged effective config and validate it before running any data processing, inference, QE scoring, external API call, or training step.

Hard rule:

```txt
If a behavior can affect data, cost, model weights, artifacts, metrics, or reproducibility, it must be config-owned.
```

---

## 2. Numeric Limits Are Config-Owned

All numeric limits must come from YAML config.

This includes:

- token limits
- source and output budgets
- subset sizes
- batch sizes
- token-budget batching limits
- retry counts
- concurrency limits
- timeouts
- training steps
- learning rates
- LoRA ranks
- selection thresholds

Implementation must not hardcode these values in source code.

Allowed:

- documenting baseline examples in `.md` files
- providing default values in checked-in YAML files
- validating relationships between numeric values

Forbidden:

- baking token limits into Python modules
- silently using magic fallback numbers when config is missing
- changing numeric behavior without changing config
- treating documentation examples as runtime source of truth

The runtime source of truth is the merged effective config.

---

## 3. Hydra Configuration

Configuration must use Hydra.

Hydra is responsible for composing split YAML files into one effective config. The effective config is the only runtime source of truth after composition.

Recommended top-level config:

```txt
configs/
  scp_stage4.yaml
  data.yaml
  inference.yaml
  training.yaml
  qe.yaml
  external_api.yaml
  logging.yaml
  prompts.yaml
  run.yaml
```

Recommended `configs/scp_stage4.yaml`:

```yaml
defaults:
  - data
  - inference
  - training
  - qe
  - external_api
  - logging
  - prompts
  - run
  - _self_
```

Rules:

- Hydra composition must happen before validation.
- The composed config must be saved as `effective_config.yaml`.
- The config hash must be computed from the composed config with secrets excluded.
- Runtime overrides must be captured by the effective config.
- Code must not read unmerged partial config files directly during pipeline execution.

---

## 4. Split Config Files

Configuration should be split by responsibility.

Recommended layout:

```txt
configs/
  data.yaml
  inference.yaml
  training.yaml
  qe.yaml
  external_api.yaml
  logging.yaml
  prompts.yaml
  run.yaml
```

The runner may merge these files into one effective config before validation.

Recommended merged artifact:

```txt
artifacts/
  runs/
    {run_id}/
      effective_config.yaml
      config_hash.txt
```

Rules:

- every run must persist the effective config
- every run must persist a config hash
- logs must include `run_id`, `subset_idx`, `phase`, and `config_hash`
- local secrets must not be written into the effective config

---

## 5. Baseline Length Config

The following is a baseline example, not a hardcoded requirement.

```yaml
model:
  name: alwaysgood/qwen35-it
  max_length: 8192
  max_seq_length: null
  dtype: bf16
  attention_impl: flash_attention_2
  padding_side: right
  eos_token: null
  chat_template: null
  trust_remote_code: false

data:
  text_columns:
    - source_text
    - text

  translatable_fields:
    - name: body
      columns:
        - source_text
        - text
      text_role: body
    - name: title
      columns:
        - metadata.Headline
        - metadata.headline
        - metadata.title
        - Headline
        - headline
        - title
      text_role: title
      optional: true

  metadata:
    document_type_field: null
    infer_document_type_from_dataset: true
    preserve_fields:
      - title
      - document_type
      - text_role

  length:
    enabled: true
    mode: tokenizer
    tokenizer_batch_size: 16384
    tokenizer_fallback: error
    max_total_tokens: 8192
    max_source_tokens: 3900
    max_output_tokens: 4096
    prompt_template_tokens: 256
    min_available_output_tokens: 768
    safety_margin_tokens: 64
    overflow: split
    output_budget_strategy: dynamic

  runtime:
    prepare_data:
      intermediate_format: parquet
      parquet_row_group_size: 4096
      progress_enabled: true
      progress_every_rows: 100000
      progress_every_seconds: 10.0
```

`min_available_output_tokens` is not a minimum generation length.

It means:

```txt
After prompt and source tokens are counted,
the row must still have at least this many tokens of possible generation space.
```

If a row leaves less than `min_available_output_tokens`, it should be split or skipped before inference.

Title/headline policy:

- `translatable_fields` may emit title/headline as separate translation rows
- title/headline should also be preserved as metadata for body rows when available
- `text_role` must be preserved so downstream prompts and logs can distinguish title vs body
- document type metadata, such as article, filing, or earnings call, should be preserved when known

---

## 6. Length Validation Rules

The validator must check:

- `model.max_length > 0`
- `model.max_seq_length` may be `null`; if null, it resolves to `model.max_length` in the effective config
- if `model.max_seq_length` is set, it must satisfy `0 < model.max_seq_length <= model.max_length`
- `data.length.max_total_tokens <= model.max_length`
- `data.length.max_total_tokens <= resolved model.max_seq_length`
- `data.length.max_source_tokens > 0`
- `data.length.max_output_tokens > 0`
- `data.length.prompt_template_tokens >= 0`
- `data.length.min_available_output_tokens > 0`
- `data.length.safety_margin_tokens >= 0`
- `data.length.tokenizer_batch_size > 0`
- `data.runtime.prepare_data.intermediate_format` must be `parquet` or `jsonl`
- `data.runtime.prepare_data.parquet_row_group_size > 0`
- `data.runtime.prepare_data.progress_enabled` must be boolean
- `data.runtime.prepare_data.progress_every_rows > 0`
- `data.runtime.prepare_data.progress_every_seconds > 0`
- `data.length.max_source_tokens + data.length.min_available_output_tokens + data.length.prompt_template_tokens + data.length.safety_margin_tokens <= data.length.max_total_tokens`

Runtime budget calculation:

```txt
available_output_budget =
  model.max_length
  - prompt_template_tokens
  - source_tokens
  - data.length.safety_margin_tokens

effective_max_new_tokens =
  min(
    data.length.max_output_tokens,
    inference.<q_tag>.max_new_tokens,
    available_output_budget
  )
```

The declared `inference.<q_tag>.max_new_tokens` is an upper bound, not a guarantee. The runtime must clamp it to `effective_max_new_tokens` per row. This prevents conflicts such as `source_tokens + prompt_tokens + max_new_tokens + safety_margin_tokens > model.max_length`.

Eligibility rule:

```txt
source_tokens <= data.length.max_source_tokens
available_output_budget >= data.length.min_available_output_tokens
```

Rows that fail eligibility must be split or skipped according to config.

---

## 7. Inference Config

The following is a baseline example.

```yaml
inference:
  q1:
    do_sample: true
    temperature: 1.1
    top_p: 0.95
    max_new_tokens: 4096
    on_truncation: discard

  q2:
    do_sample: false
    temperature: 0.0
    top_p: null
    max_new_tokens: 4096
    on_truncation: discard

  throughput:
    batching:
      strategy: token_budget
      max_batch_tokens: 32768
      pad_to_multiple_of: 8
    preserve_order: false
    restore_order_in_artifacts: true
```

Q1/Q2 throughput settings control how many rows are generated together during `infer-q1` and `infer-q2`.

Rules:

- batching must respect token budgets, not only row counts
- long rows should not force excessive padding for short rows
- row ids must remain deterministic under reordering
- artifacts must be restorable to original row order
- if generated output hits the effective limit, handle it using `on_truncation`

---

## 8. Pipeline Config

Baseline example:

```yaml
pipeline:
  subset:
    strategy: fraction
    fraction: 0.02        # 1/50 of train datapool
    min_size: 32
    max_size: null
    shuffle: true
    seed: 42
    drop_last: false

  stage:
    max_subsets: null
    use_sampled_data: false
    subset_archive:
      enabled: false
      format: tar.gz
      output_dir: archives/subsets
      delete_original_after_archive: false

  execution:
    allow_microbatch_overlap: true
    allow_next_subset_q1_prefetch: false
    api_concurrency_overlaps_gpu_work: true

  eval_after_subset:
    enabled: true
    dataset: ood_test
    every_n_subsets: 1
    run_on_final_subset: true
    runtime: qe_subprocess
    source_column: Source_En
    reference_column: Target_Ko
    metrics:
      - metricx24_ref
      - BLEU
      - chrF
    metric_settings:
      BLEU:
        library: sacrebleu
        level: sentence
        effective_order: true
        smooth_method: exp
      chrF:
        library: sacrebleu
        level: sentence
        word_order: 2
```

Rules:

- subset size may be computed as `ceil(train_datapool_size * fraction)`
- `fraction: 0.02` means roughly 50 subsets per full datapool pass
- `subset_idx` is the SCP iteration index
- `allow_microbatch_overlap` permits overlapping API calls for completed selected microbatches with Q1/Q2 work for other microbatches inside the same subset
- `allow_next_subset_q1_prefetch` defaults to `false` because Q1 for the next subset must use the base model after the current subset update
- OOD eval runs after every subset by default
- OOD eval after subset uses the configured `ood_test` set, not the training datapool
- `Target_Ko` is the reference column for OOD reference-based metrics
- `runtime: qe_subprocess` means MetricX, BLEU, and chrF are computed outside the main training process
- BLEU and chrF settings must match the reference notebook unless explicitly changed in config
- `stage.subset_archive.enabled` controls whether completed subset directories are archived as one file per subset
- `stage.subset_archive.format` must be one of `tar`, `tar.gz`, `tar.xz`
- `stage.subset_archive.output_dir` is relative to `artifacts/runs/{run_id}`
- `stage.subset_archive.delete_original_after_archive` may prune subset directories after stage completion; archive and manifest must exist first

---

## 9. Training Config

All SCP training must use Unsloth.

Baseline example:

```yaml
training:
  backend: unsloth

  collapse_lora:
    rank: 4
    learning_rate: 0.005
    num_train_epochs: 1
    dropout: 0.0
    bias: none

  base_update:
    mode: lora       # lora | full_weight
    persistence: cumulative
    save_after_each_subset: true
    num_train_epochs: 1

    lora:
      rank: 32
      alpha: 64
      dropout: 0.0
      bias: none
      use_rslora: false
      loftq_config: null
      target_modules:
        - q_proj
        - k_proj
        - v_proj
        - o_proj
        - gate_proj
        - up_proj
        - down_proj
        - in_proj_a
        - in_proj_b
        - in_proj_z
        - in_proj_qkv
        - out_proj

    optimizer:
      learning_rate: 1.0e-5
      weight_decay: 0.0
      warmup_ratio: 0.1
      lr_scheduler_type: cosine
      optim: adamw_torch
      max_grad_norm: 1.0

    batching:
      per_device_train_batch_size: 8
      gradient_accumulation_steps: 4
      per_device_eval_batch_size: 8
      packing: false
      response_template: null

    data:
      use_no_change: true

  checkpoint:
    save_after_each_subset: true
    save_latest_pointer: true
    keep_subset_checkpoints: true
    keep_last_n: 2
    keep_best_n: 2
    metric_for_best: ood/metricx24_ref_quality_mean
    greater_is_better: true
    keep_final: true
    save_optimizer_state: true
    save_collapse_lora: false
    upload_to_wandb: false

  eval:
    internal:
      enabled: true
      strategy: steps
      eval_steps: 50
      eval_on_start: false
```

Rules:

- `training.backend` must be `unsloth`
- TRL `SFTTrainer` may be used only with Unsloth-prepared models
- plain Hugging Face Trainer or PEFT-only paths that bypass Unsloth are forbidden
- collapse LoRA and base update both run for one epoch unless config changes
- collapse LoRA must not be merged into the base model
- base update is cumulative across subsets
- `base_update.mode: full_weight` must use the same optimizer/batching schema unless overridden
- `training.base_update.lora.target_modules` may be either:
  - a string shortcut such as `all-linear`
  - a list of module names (for example attention/MLP + DeltaNet projections)
- collapse LoRA is not saved by default
- base update checkpoint is saved after every subset
- checkpoint retention must prevent unbounded disk growth
- OOD eval is the default subset-completion evaluation target
- `pipeline.eval_after_subset` is the only authoritative config location for subset-level OOD eval
- `training.eval` is only for internal trainer eval during base-update training
- model loading keys such as `chat_template`, `eos_token`, `dtype`, `padding_side`, and `attention_impl` must be explicit in config or explicitly resolved by the model loader and written to the effective config

---

## 10. QE Config

Baseline example:

```yaml
qe:
  epsilon: 1.0e-6

  primary:
    backend: metricx24
    model_name: google/metricx-24-hybrid-xxl-v2p6-bfloat16
    tokenizer_name: google/mt5-xl
    batch_size: 8
    max_input_length: 1536
    score_direction: lower_is_better
    transform:
      type: invert
      max_score: 25.0
      clamp_for_quality: true
      preserve_raw_score: true

  alternatives:
    - backend: comet_kiwi
      model_name: Unbabel/wmt23-cometkiwi-da-xl
      fallback_model_name: Unbabel/wmt22-cometkiwi-da
      batch_size: 8
      gpus: 1
      score_direction: higher_is_better
      transform:
        type: none

  scoring:
    weighted_score:
      enabled: true
      alpha: 0.3
      beta: 0.7
    collapse_term:
      type: c1
      collapse_rate_threshold: 0.0
    standardization:
      method: zscore
      scope: batch
    selection:
      policy: configurable
      default_rule:
        require_score_s_gte: 0.0
        top_fraction: 0.10
      rules: []

  isolation:
    enabled: true
    setup:
      managed_install: false
      runtime: metricx      # metricx | comet | both | skip
      shared_venv_allowed: true
      cuda_wheel_detection: nvidia_smi
    env:
      comet_python_env: COMET_PYTHON
      metricx_python_env: METRICX_PYTHON

  failure:
    retry: 1
    on_row_error: skip
    on_backend_error: fail
```

QE failures include:

- QE virtual environment missing
- QE model load failure
- CUDA OOM during QE scoring
- subprocess timeout
- malformed JSONL input or output
- non-JSON progress output written to stdout
- NaN or invalid score for one or more rows

Rules:

- row-level failures may be skipped and logged
- backend-level failures must fail the run because selection cannot be trusted
- all QE scoring config lives under `qe.*`; do not introduce `qe_scoring.*`
- QE drivers return raw backend scores; the scoring layer owns transform, clamping, and interpretation
- `qe.alternatives` is an object list using the same backend schema as `qe.primary`
- selection policy is intentionally configurable; baseline examples may use `score_s >= 0` plus top fraction, but the exact selection expression is experiment-owned
- dummy QE is allowed only for explicit tests or PoC dry runs
- `managed_install: false` means the runtime expects `COMET_PYTHON` and/or `METRICX_PYTHON` to point to an existing venv
- `shared_venv_allowed: true` allows both env vars to point to the same Python binary, as in the PoC setup
- QE runtime details belong in `docs/qe-isolation.md`

Subprocess worker CLI contract:

- when `*.runtime.mode: subprocess`, the runner calls workers with:
  - `--input` and `--output`
  - `--effective-config` and `--config-hash`
  - `--run-id`, `--subset-idx`, `--section`, `--phase`
- workers should validate required request fields and required response fields per phase before writing output JSONL
- missing required response keys must fail the worker with non-zero exit code
- checked-in real profile: `configs/scp_stage4_real.yaml`
  - inference worker: `python3 -m scp_stage4.pipeline.workers.inference_worker`
  - QE worker: `python3 -m scp_stage4.pipeline.workers.qe_worker`
  - external API worker: `python3 -m scp_stage4.pipeline.workers.external_api_worker`
  - training worker: `python3 -m scp_stage4.pipeline.workers.training_worker`

---

## 11. External API Config

Baseline example:

```yaml
external_api:
  routing:
    mode: single       # single | fallback | ensemble
    fallback_enabled: false
    ensemble_enabled: false

  primary:
    provider: openai
    model: gpt-5.4  # placeholder; replace with an available provider model before real API calls
    api_key_env: OPENAI_API_KEY

  providers:
    openai:
      enabled: true
      api_key_env: OPENAI_API_KEY
    gemini:
      enabled: false
      api_key_env: GEMINI_API_KEY
    claude:
      enabled: false
      api_key_env: ANTHROPIC_API_KEY
    qwen:
      enabled: false
      api_key_env: QWEN_API_KEY
    deepseek:
      enabled: false
      api_key_env: DEEPSEEK_API_KEY

  failure:
    retry: 2
    on_error: skip
    max_failure_rate: 0.2
    on_failure_rate_exceeded: fail

  output_status:
    allowed:
      - ok
      - skipped
      - failed
      - filtered
      - needs_review
```

Rules:

- secrets must be referenced by environment variable name only
- API keys must never be written to config artifacts or logs
- fallback and ensemble schemas may exist, but default routing is `single`
- external API calls happen only after selection
- placeholder model names are allowed in docs/config examples but must fail validation before real API execution unless replaced with an available provider model
- correction prompt details belong in `docs/prompts.md`
- provider request/response schemas belong in `docs/external-api.md`

---

## 12. Logging Config

Baseline example:

```yaml
logging:
  report_to:
    - wandb

  run_name: null

  local:
    enabled: true
    root_dir: artifacts/runs
    write_effective_config: true
    write_config_hash: true

  wandb:
    enabled: true
    project: scp_main
    entity: null
    tags:
      - scp
      - stage4
      - sft
      - en-ko
    notes: ""
    log_artifacts: true
    log_model_checkpoints: false

  weave:
    enabled: true
    project: scp_main
    print_call_link: false
    implicitly_patch_integrations: true
    trace_external_api: true
    trace_qe_subprocess: false
    redact_inputs: false
```

Rules:

- W&B and Weave project should default to `scp_main`
- local JSONL artifacts remain the source of truth
- W&B is for experiment metrics and artifact tracking
- Weave is for external API/application traces
- secrets must not be logged or written to config artifacts
- detailed logging contracts belong in `docs/logging.md`

---

## 13. Agent Notes

- This schema document defines ownership and validation rules, not final experimental constants.
- Numeric values shown here are baseline YAML examples.
- Future experiments should change config files, not source code, when adjusting limits.

## Suggested Improvements

- Implement the harness validation foundation first: Makefile targets, Hydra config loading, and JSONL schema checking.
- Add `make validate-local`, `make test-local`, and `make smoke-local` for local contract validation.
- Add remote validation targets for GPU/API/QE checks: `validate-remote-env`, `smoke-remote-qe`, `smoke-remote-model`, `smoke-remote-api`, and `dry-run-remote-subset`.
- Add tests that fail when required numeric limits are missing.
