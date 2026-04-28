# docs/training.md

# SCP Stage 4 Training

> **Project**: `scp_stage4_sft`  
> **Base Model**: `alwaysgood/qwen35-it`  
> **Scope**: Collapse LoRA, base update training, LoRA lifecycle, optimizer design, and training flow.

---

# 1. Overview

SCP Stage 4 uses two distinct training processes:

```txt
1. Collapse LoRA (temporary, probing)
2. Base Update (persistent, improvement)
```

These MUST remain strictly separated.

---

# 2. Training Flow (Subset-Level)

Within one subset:

```txt
Q1 inference (base model)
  ↓
train collapse LoRA (pseudo-label)
  ↓
Q2 inference (with collapse LoRA)
  ↓
score and select fragile samples
  ↓
unload collapse LoRA
  ↓
external API generates gold
  ↓
base update training
```

---

# 3. Base Model

```txt
alwaysgood/qwen35-it
```

Assumptions:

- instruction-tuned causal LM
- supports LoRA via Unsloth
- transformer-based architecture with attention + MLP blocks

Required model loading config:

```yaml
model:
  name: alwaysgood/qwen35-it
  max_length: 8192
  max_seq_length: null
  dtype: bf16
  attention_impl: flash_attention_2
  padding_side: right
  chat_template: null
  eos_token: null
  trust_remote_code: false
```

Rules:

- chat template, EOS behavior, dtype, padding side, attention implementation, and max sequence length must be explicit in config or resolved by the loader and written to the effective config
- `model.max_seq_length: null` resolves to `model.max_length`; if explicitly set, it must be less than or equal to `model.max_length`
- SFT prompt formatting must be reproducible from prompt config and model loading config

---

# 4. Collapse LoRA (CRITICAL)

## Purpose

- induce controlled degradation
- reveal fragile samples via Q1 → Q2 difference

---

## Configuration

```yaml
collapse_lora:
  rank: 4
  learning_rate: 0.005
  num_train_epochs: 1
  dropout: 0.0
  bias: none
```

---

## Target Modules

### Attention

```txt
q_proj
k_proj
v_proj
o_proj
```

### MLP

```txt
gate_proj
up_proj
down_proj
```

---

## Training Data

- input: `source`
- label: `mt_q1` (pseudo-label)

---

## Rules

- must NOT be merged into base model
- must be unloaded after Q2 inference
- must NOT persist across subsets
- one collapse adapter per subset

---

## Output

```txt
collapse_adapter/
```

This path is only for explicit debugging. Baseline runs do not persist collapse LoRA checkpoints.

---

## Collapse Strength Warning

Including MLP layers increases collapse strength.

Baseline collapse LoRA targets attention and MLP modules. If collapse becomes globally destructive, removing MLP modules is a mitigation, not the default.

### Symptoms of excessive collapse

- Q2 is globally degraded
- all samples appear equally bad
- scoring becomes noisy

### Mitigation

- reduce LR (0.005 → 0.002)
- reduce rank (4 → 2)
- remove MLP modules
- reduce training steps

---

# 5. Base Update Training

## Purpose

- improve translation quality
- incorporate external LLM corrections

---

## Input

- `source`
- `gold` (from external API)

---

## Modes

```yaml
base_update:
  mode: lora   # default
```

---

### Option A: LoRA (Default)

```yaml
base_update:
  mode: lora
  persistence: cumulative
  num_train_epochs: 1

  lora:
    rank: 32
    alpha: 64
    dropout: 0.0
    bias: none
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - gate_proj
      - up_proj
      - down_proj

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
```

Advantages:

- fast iteration
- memory efficient
- stable training

---

### Option B: Full Weight

```yaml
base_update:
  mode: full_weight
  num_train_epochs: 1

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
```

Advantages:

- maximum performance potential

Tradeoffs:

- higher cost
- risk of overfitting
- slower iteration

---

## Persistence

- updates affect all future subsets
- checkpoint must be saved after each subset
- collapse LoRA is not saved by default
- base update checkpoints must use a retention policy to avoid unbounded disk growth

---

## Checkpoint Strategy

Recommended layout:

```txt
artifacts/
  runs/
    {run_id}/
      checkpoints/
        latest -> ../subsets/subset_004/train_final
        best/
          subset_002/
          subset_004/

      subsets/
        subset_000/
          train_final/
            main_adapter/
            tokenizer/
            trainer_state.json
            train_metrics.json
            eval_metrics.json
```

Baseline config:

```yaml
training:
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
```

Rules:

- save base update after every subset
- keep a `latest` pointer for resume
- keep immutable checkpoints for the last `N` subsets
- keep best checkpoints according to configured OOD metric
- keep final checkpoint even if it is not in last/best retention sets
- delete or archive older checkpoints according to retention config
- do not save collapse LoRA unless explicitly debugging
- W&B checkpoint upload is disabled by default to avoid cost and storage growth

---

## Eval Strategy

There are two eval types.

### Internal Training Eval

Purpose:

- monitor training loss
- detect overfitting or instability during base update

Baseline config:

```yaml
training:
  eval:
    internal:
      enabled: true
      strategy: steps
      eval_steps: 50
      eval_on_start: false
```

### OOD Eval After Subset

Purpose:

- measure actual English-to-Korean quality after subset update
- track stage-level progress on the held-out OOD test set

Authoritative config lives in `pipeline.eval_after_subset`, not `training.eval`.

Baseline config reference:

```yaml
pipeline:
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

- OOD eval uses the configured `ood_test` artifact
- OOD eval runs after every subset by default
- `Target_Ko` is the OOD reference column
- MetricX-24 reference-based scoring should use `Source_En`, model hypothesis, and `Target_Ko`
- BLEU and chrF must be recorded alongside the neural reference metric
- BLEU and chrF must match the reference notebook's sacrebleu settings
- MetricX, BLEU, and chrF should run through the isolated QE/eval subprocess
- OOD eval rows are never used for training or API correction
- full runs may override `every_n_subsets` only when runtime cost requires it
- do not duplicate this config under `training.eval.after_subset`

---

# 6. LoRA Lifecycle (CRITICAL)

```txt
Q1 → collapse LoRA → Q2 → score/select → unload → API correction → base update
```

Rules:

- collapse LoRA must never contaminate base
- collapse LoRA must be removed immediately after Q2 artifacts are produced and before API correction or base update
- base model must be verified clean

---

## Required Check

```python
assert_clean_base()
```

Minimum checks:

- no active collapse adapter is present in the model adapter registry
- active adapter is either `null` or the configured main/base-update adapter
- collapse adapter weights are not merged into base weights
- generated trainable parameter names do not include the collapse adapter namespace
- optional weight hash or adapter-state hash check passes when configured
- a small clean-base generation smoke test can run without referencing the collapse adapter

Required subset artifact (`clean_base.json`) fields:

```json
{
  "status": "ok",
  "run_id": "run_id",
  "subset_idx": 0,
  "clean_base": true,
  "active_adapters": [],
  "collapse_merged": false,
  "adapter_registry_hash": "sha256-like string",
  "verified_adapter_path": "path/to/subset/collapse_adapter",
  "status_rows": [
    {
      "status": "ok",
      "clean_base": true,
      "active_adapters": [],
      "collapse_merged": false,
      "adapter_registry_hash": "sha256-like string",
      "verified_adapter_path": "path/to/subset/collapse_adapter"
    }
  ]
}
```

Additional contract rules:

- `clean_base` must be `true`
- `active_adapters` must be an empty list
- `collapse_merged` must be `false`
- `adapter_registry_hash` must be non-empty and generated by the unload worker
- for subprocess training runtime, missing clean-base evidence fields must fail fast

---

# 7. Optimizer

## Collapse LoRA

```yaml
optimizer:
  type: AdamW
  learning_rate: 0.005
  weight_decay: 0.0
```

- high LR is intentional (for collapse)

---

## Base Update

```yaml
optimizer:
  type: AdamW
  learning_rate: 1.0e-5
  weight_decay: 0.0
  warmup_ratio: 0.1
```

Optional:

```yaml
scheduler:
  type: cosine
```

---

# 8. Training Constraints

- config-driven only
- subset-level execution
- checkpoint required
- must support resume
- must not depend on QE or API modules

---

# 9. Logging Requirements

Each training step must log:

```yaml
run_id
subset_idx
phase
config_hash
loss
learning_rate
num_steps
```

Collapse-specific:

```yaml
collapse_strength
```

---

# 10. Failure Modes

## Collapse Failures

| Symptom | Cause |
|--------|------|
| Q1 ≈ Q2 | collapse too weak |
| Q2 broken globally | collapse too strong |

---

## Base Update Failures

| Symptom | Cause |
|--------|------|
| no improvement | weak signal |
| regression | overfitting |
| unstable loss | LR too high |

---

# 11. Implementation Notes

- use Unsloth LoRA
- separate adapter names:
  - `collapse_adapter`
  - `main_adapter`
- do not mutate global model state
- keep loops minimal

---

# 12. Design Principle

> Collapse reveals weakness.  
> Update fixes it.

Never mix these two roles.

---

# 13. Training Backend (Unsloth)

All training in this project MUST use **Unsloth**.

TRL `SFTTrainer` is allowed only when the model has been loaded, patched, and prepared through Unsloth.

Both collapse LoRA and base update MUST use Unsloth.

---

## Why Unsloth

- optimized for LLM fine-tuning
- faster than standard Hugging Face Trainer
- lower memory usage
- strong support for Qwen models
- compatible with TRL `SFTTrainer` when the model is prepared through Unsloth

---

## Collapse LoRA with Unsloth

- implemented using Unsloth LoRA API
- temporary adapter
- same constraints as defined above

---

## Base Update with Unsloth

Supported modes:

- LoRA (default)
- full-weight training (optional)

---

## Rules

- do not use plain Hugging Face Trainer or PEFT-only training paths that bypass Unsloth
- TRL `SFTTrainer` may be used as the training loop for Unsloth-prepared models
- do not fallback silently
- training backend must be config-driven

---
