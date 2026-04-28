# AGENTS.md

> **Project**: `scp_stage4_sft`  
> **Purpose**: Define the agent harness for implementing and maintaining the SCP Stage 4 English→Korean translation pipeline.  
> **Important**: Every agent must read this file before making changes.

---

# 0. Start Here (Strict)

If you are a new agent:

1. Read:
   - `docs/scp-overview.md`
   - `docs/data-pipeline.md`
   - `docs/config-schema.md`

2. Validate configuration:
   ```bash
   make validate-config
   ```

3. Use a small subset for testing:
   ```yaml
   data:
     subset_size: 32
   ```

4. Do NOT start coding before understanding:
   - subset iteration loop
   - collapse vs update distinction
   - data artifacts and flow

---

# 1. Project Goal

This repository implements **Self-Collapse Probing (SCP)** to improve English→Korean translation.

Core idea:

- identify fragile samples via model collapse
- correct them using external LLM
- iteratively update the base model

---

# 2. Pipeline Mental Model (CRITICAL)

This is an **iterative subset-based pipeline**, not a single-pass training loop.

Each subset performs:

```txt
subset_i
  ↓
Q1 inference (base model)
  ↓
train collapse LoRA (pseudo labels)
  ↓
Q2 inference (with collapse LoRA)
  ↓
score S_i(Q1, Q2)
  ↓
select fragile samples
  ↓
unload collapse LoRA
  ↓
external API correction (gold)
  ↓
update base model
  ↓
next subset
```

Key properties:

- Collapse LoRA is temporary and used only for probing
- Base model is updated after each subset
- Subsets are processed sequentially
- Each subset affects future subsets

---

# 3. Execution Interface (Makefile)

Agents MUST use Makefile targets.

`make set` is a lightweight local bootstrap target. It should install or verify editable/local development dependencies and create required local directories. It must not require GPU, download large models, call external APIs, or verify CUDA kernels. Remote/runtime setup belongs in explicit remote targets.

## Core commands

```bash
make set
make validate-config
make validate-local
make test-local
make smoke-local
make prepare-data
make run-subset
make run-stage
make eval
make eval-ood
```

## Remote validation commands

These targets are intended for the external GPU/API instance, not necessarily the local development machine.

```bash
make validate-remote-env
make smoke-remote-qe
make smoke-remote-model
make smoke-remote-api
make dry-run-remote-subset
```

## Subset-level steps

```bash
make infer-q1
make train-collapse-lora
make infer-q2
make score
make call-api
make update-base
```

---

# 4. Step Definitions

| Step | Description |
|------|-------------|
| infer-q1 | Generate `mt_q1` with the base model and compute/store `qe_q1` when configured |
| train-collapse-lora | Train temporary weak LoRA (collapse probing) |
| infer-q2 | Generate `mt_q2` with the collapse LoRA and compute/store `qe_q2` when configured |
| score | Compute S_i and select fragile samples |
| unload-collapse-lora | Remove temporary collapse adapter and verify base model is clean |
| call-api | Generate corrected translations using external LLM |
| update-base | Update main model using corrected data |

---

# 5. Training Types (CRITICAL)

There are TWO distinct training processes.

## 5.1 Collapse LoRA (temporary)

- purpose: observe model fragility
- uses pseudo labels (Q1)
- weak configuration (e.g. rank=4, lr=0.005)
- must NOT be merged into base
- must be unloaded after use

---

## 5.2 Base Update (persistent)

- purpose: improve translation quality
- uses corrected (gold) data
- may be:
  - LoRA tuning (main adapter)
  - full-weight fine-tuning
- persists across subsets

---

# 6. LoRA Lifecycle Rules

After Q2 inference and before any external API or base-update work:

- collapse LoRA MUST be unloaded
- base model MUST be clean

Never:

- merge collapse LoRA into base
- reuse collapse LoRA across subsets

---

# 7. Data & Artifacts

All intermediate data must be explicit and reproducible.

```txt
artifacts/
  data/
    datapool.train.jsonl
    datapool.eval.jsonl
    ood_test.jsonl

  runs/
    {run_id}/
      subsets/
        subset_000/
          input.jsonl
          q1.jsonl
          q2.jsonl
          scored.jsonl
          selected.jsonl
          api_requests.jsonl
          api.jsonl
          train_final/
```

Each subset must be reproducible from:

- config
- seed
- dataset artifact
- base checkpoint

---

# 8. Data Rules

- one row = one sample
- minimal filtering
- tokenizer-based length control
- overflow handled via sentence-aware split
- eval_ratio default = 0.02
- OOD test set is strictly evaluation-only

---

# 9. Hard Rules

## YAML-only configuration

- no hardcoding
- missing config → fail

---

## QE isolation

- COMET / MetricX must run in separate venv
- subprocess only
- JSONL I/O only
- never import in main runtime

---

## External API

- only call after selection
- log cost, tokens, latency
- never log secrets

---

## Logging

Must include:

```yaml
run_id
subset_idx
phase
config_hash
```

---

## Throughput

The SCP loop can be slow because each subset performs generation, collapse training, QE scoring, external API correction, and base update training.

Implementations must be designed for high throughput:

- cache token counts and prompt lengths
- bucket rows by token length
- batch by token budget, not only row count
- avoid avoidable serial bottlenecks
- use configurable concurrency for QE and external API calls
- preserve deterministic row ids under batching, bucketing, and retries
- keep all throughput behavior YAML-configured

Never improve throughput by dropping required artifacts, weakening reproducibility, or hiding failures.

---

## Implementation Feedback Loop

Coding happens locally, while real GPU/API/QE validation happens on an external instance. Local validation is intentionally lightweight: it checks deterministic logic and file/interface contracts, not real training quality or runtime performance.

Coding must proceed through small, verifiable slices.

For each new module or pipeline step, agents must:

1. define the input/output contract before implementation
2. add or update the smallest relevant config schema
3. implement the narrow step
4. run the narrow validation target
5. check artifact shape and row counts
6. check compatibility with adjacent steps
7. only then connect it to the full subset loop

Local required checks:

- config validation after config/schema changes
- unit or smoke tests for each step
- JSONL schema validation for produced fixture artifacts
- row-id preservation across input, q1, q2, scored, selected, api, and train artifacts
- mocked QE subprocess contract when real QE is unavailable
- mocked external API contract when real API calls are unavailable
- tiny fixture smoke test that does not require GPU, real QE model, or real API calls

Remote required checks:

- remote environment validation
- real QE venv availability
- real model load smoke test
- real Q1/Q2 generation smoke test
- collapse LoRA training smoke test
- external API smoke test with a tiny selected set
- dry-run subset with `data.subset_size: 32`
- one end-to-end `make run-subset` before `make run-stage`

Completion rule:

- local implementation completion requires lightweight logic/contract validation
- remote readiness requires real dry-run subset validation on the external instance

Failure handling:

- fail fast on schema mismatch, missing config, missing artifact, or row-id drift
- retry only configured transient failures such as external API timeout
- do not silently drop rows; mark them with explicit status
- log enough context to reproduce the failing step
- keep fixes local to the failing contract unless a cross-file interface change is required

---

## External Library Rule

When implementing code that uses external libraries, agents must first consult the official documentation for the installed or configured version.

This applies especially to:

- Unsloth
- TRL `SFTTrainer`
- Hydra / OmegaConf
- W&B
- Weave
- MetricX
- COMET / COMET-Kiwi
- sacrebleu
- OpenAI / external LLM provider SDKs

Rules:

- prefer official docs or source examples over memory
- if official docs are unclear, inspect well-known open-source implementations
- record important version-sensitive assumptions in docs or comments
- do not invent library APIs
- do not copy notebook code directly into `src`
- keep library-specific behavior behind small wrapper modules when possible

---

## Training Backend

All training must use **Unsloth**.

TRL `SFTTrainer` is allowed only when the model has been loaded, patched, and prepared through Unsloth.

- Collapse LoRA must use Unsloth LoRA
- Base update must use Unsloth LoRA or Unsloth full-weight training
- Do not use plain Hugging Face Trainer or PEFT-only training paths that bypass Unsloth

---

# 10. Environment Setup

```bash
git clone https://github.com/contiloop/scp_stage4_sft.git
cd scp_stage4_sft
make set
```

Login:

```bash
python -c "from huggingface_hub import login; login()"
wandb login
```

Rules:

- never hardcode tokens
- never commit secrets
- QE env must be separate

---

# 11. Workflow

## One subset

```bash
make run-subset
```

## One stage

```bash
make run-stage
```

## Dry run

Local dry run:

```bash
make validate-local
make test-local
make smoke-local
```

Remote dry run:

```bash
make validate-config
make prepare-data
make dry-run-remote-subset
```

## Makefile Target Contracts

Every Makefile target must document:

- required config keys
- input artifacts
- output artifacts
- whether GPU/API/QE runtime is required
- expected exit code behavior

Exit code contract:

- `0`: target completed and produced the declared outputs
- non-zero: target failed and must write a structured failure event when logging is available
- validation targets must fail on missing config, schema mismatch, missing artifact, or row-id drift

The first implementation milestone is to create these target contracts for local validation before implementing full training.

---

# 12. Failure Isolation

| Symptom | Cause |
|--------|------|
| QE NaN | QE subprocess |
| Q1 ≈ Q2 | collapse failure |
| Q2 collapse everywhere | too strong collapse |
| no selection | scoring issue |
| API noisy | prompt issue |
| regression | update-base issue |

---

# 13. Documentation Routing

| Task | Read |
|------|------|
| data | docs/data-pipeline.md |
| training | docs/training.md |
| QE | docs/qe-scoring.md |
| API | docs/external-api.md |
| logging | docs/logging.md |
| config | docs/config-schema.md |

---

# 14. Agent Feedback Loop

Every task must end with:

```md
## Agent Notes
- unclear parts
- assumptions
- failures

## Suggested Improvements
- docs
- config
- scripts
```

---

# 15. Forbidden

- no monolithic pipeline
- no QE in main process
- no collapse LoRA merge
- no API before selection
- no secrets in code or logs
- no notebook copy into src

---

# 16. Principle

> This is not just a training script.  
> This is an **agent-executable SCP system**.

Focus on:

- reproducibility
- modularity
- clear contracts
- failure isolation
