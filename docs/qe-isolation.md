# SCP Stage 4 QE Isolation

> **Project**: `scp_stage4_sft`  
> **Scope**: QE runtime isolation, virtual environments, subprocess execution, JSONL I/O, dependency separation, and backend runtime parameters.

---

## 1. Goal

QE models must run outside the main SCP training process.

This document defines:

- how QE runtimes are isolated from the main training runtime
- how COMET-Kiwi and MetricX are executed
- how subprocess JSONL input/output works
- how backend runtime parameters are configured
- how dependency conflicts are avoided
- how QE failures are isolated and reported

This document does **not** define:

- Q1 / Q2 semantics
- MetricX score direction conversion
- collapse terms
- weighted `S_i`
- sample selection

Those belong in:

```txt
docs/qe-scoring.md
```

Boundary rule:

```txt
QE isolation runs the backend model.
QE scoring interprets the backend score.
```

---

## 2. Why Isolation Is Required

QE must be isolated because the QE stack and the training stack may require incompatible dependencies.

Common conflict pattern:

```txt
main training environment
  - Unsloth / training dependencies
  - training-specific transformers version
  - LoRA / base model runtime

QE environment
  - unbabel-comet
  - MetricX dependencies
  - potentially different transformers version
```

Hard rules:

- the main runtime must never import COMET directly
- the main runtime must never import MetricX model code directly
- QE inference must happen through subprocess calls
- QE input/output must use JSONL
- QE failure must not silently corrupt the SCP run

---

## 3. Core Execution Model

```txt
main SCP process
    |
    |  JSONL request file or JSONL stdin
    v
QE subprocess running in QE venv
    |
    |  load QE backend
    |  run batched inference
    v
JSONL response file or JSONL stdout
```

The main SCP process is responsible for:

- preparing JSONL requests
- launching the subprocess with the correct Python binary
- parsing JSONL responses
- mapping returned scores back to row ids
- handling subprocess failures

The QE subprocess is responsible for:

- importing backend-specific dependencies
- loading the QE model
- running inference
- returning raw backend scores
- writing logs to stderr, not stdout

---

## 4. Runtime Discovery

The main process discovers QE Python runtimes through environment variables.

```bash
COMET_PYTHON=/path/to/comet-venv/bin/python
METRICX_PYTHON=/path/to/metricx-venv/bin/python
```

Rules:

- COMET-Kiwi backends use `COMET_PYTHON`
- MetricX backends use `METRICX_PYTHON`
- both variables may point to the same QE venv if dependencies are compatible
- separate backend-specific venvs may be used if dependencies conflict
- production runs must fail if the required runtime variable is missing

Example:

```bash
export COMET_PYTHON="$HOME/.venvs/comet/bin/python"
export METRICX_PYTHON="$HOME/.venvs/comet/bin/python"
```

The PoC notebook used one shared venv at `~/.venvs/comet` for both COMET and MetricX and pointed both variables to the same Python binary. This is allowed when dependencies are compatible.

---

## 5. Virtual Environment Policy

QE dependencies must be isolated from the main training environment.

Recommended PoC layout:

```txt
~/.venvs/
  comet/
    bin/python
```

Recommended production layout if dependency conflicts appear:

```txt
~/.venvs/
  comet/
    bin/python

  metricx/
    bin/python
```

Allowed:

- one shared QE venv for COMET and MetricX, if compatible
- separate COMET and MetricX venvs, if required
- using `~/.venvs/comet` as the shared PoC venv even when MetricX is the active backend

Forbidden:

- installing COMET into the main training runtime
- importing QE libraries from the main process
- relying on notebook state
- hardcoding local absolute paths in source code
- hardcoding Hugging Face tokens in notebooks, configs, scripts, or logs

---

## 6. Authentication Assumption

Hugging Face and W&B authentication are handled during project setup.

This document assumes setup has already run:

```bash
make set
python -c "from huggingface_hub import login; login()"
wandb login
```

QE subprocesses should inherit authentication from the environment.

Allowed inherited variables:

```txt
HF_TOKEN
HUGGINGFACE_HUB_TOKEN
CUDA_VISIBLE_DEVICES
WANDB_API_KEY
```

Rules:

- never hardcode tokens
- never commit secrets
- never write secrets to logs
- never require tokens in YAML config

---

## 7. Backend Runtime Parameters

Runtime parameters belong in the runtime/isolation layer, not in `qe-scoring.md`.

Recommended config:

```yaml
qe_runtime:
  metricx24:
    batch_size: 8
    max_input_length: 1536
    device: cuda

  comet_kiwi:
    batch_size: 8
    gpus: 1
```

Meaning:

| Field | Meaning |
|---|---|
| `batch_size` | QE inference batch size |
| `max_input_length` | tokenizer/model input limit for MetricX |
| `device` | runtime device for MetricX |
| `gpus` | COMET `model.predict(..., gpus=N)` argument |

Rules:

- these parameters affect memory, speed, and runtime stability
- these parameters must not change the definition of Q1, Q2, collapse, or `S_i`
- missing runtime config must fail validation

---

## 8. MetricX Runtime Setup

MetricX runtime should be installed inside a QE venv.

Example:

```bash
python3 -m venv ~/.venvs/comet

~/.venvs/comet/bin/python -m pip install -U pip "setuptools<82" wheel
~/.venvs/comet/bin/python -m pip install torch torchvision torchaudio
~/.venvs/comet/bin/python -m pip install transformers sentencepiece safetensors accelerate huggingface_hub
```

If CUDA-specific PyTorch wheels are required, use the matching PyTorch index.

Example:

```bash
~/.venvs/comet/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch torchvision torchaudio
```

The PoC setup script detected the host CUDA runtime with `nvidia-smi` and chose the closest supported PyTorch wheel index such as `cu128`. Production setup scripts may follow the same pattern, but the chosen index must be logged.

Supported MetricX model:

```txt
google/metricx-24-hybrid-xxl-v2p6-bfloat16
```

MetricX driver returns raw MetricX backend scores.

Score direction conversion is handled in `docs/qe-scoring.md`.

---

## 9. COMET Runtime Setup

COMET runtime should be installed inside a QE venv.

Example:

```bash
~/.venvs/comet/bin/python -m pip install "unbabel-comet>=2.2.7" huggingface_hub
```

Supported COMET-Kiwi models:

```txt
Unbabel/wmt23-cometkiwi-da-xl
Unbabel/wmt22-cometkiwi-da
```

Reference COMET usage pattern:

```python
from comet import download_model, load_from_checkpoint

model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
model = load_from_checkpoint(model_path)

data = [
    {
        "src": "English source text",
        "mt": "Korean translation"
    }
]

model_output = model.predict(data, batch_size=8, gpus=1)
```

The COMET driver returns raw COMET backend scores.

Score interpretation is handled in `docs/qe-scoring.md`.

---

## 10. JSONL Input Contract

QE subprocess input must be JSONL.

Each line represents one translation candidate to score.

Example:

```jsonl
{"id":"run_abc123/subsets/subset_000/sample_000001/q1","run_id":"run_abc123","subset_idx":0,"row_id":"sample_000001","q_tag":"q1","backend":"metricx24","src":"English source text","mt":"Korean translation"}
{"id":"run_abc123/subsets/subset_000/sample_000001/q2","run_id":"run_abc123","subset_idx":0,"row_id":"sample_000001","q_tag":"q2","backend":"metricx24","src":"English source text","mt":"Korean translation after collapse LoRA"}
```

Required fields:

| Field | Description |
|---|---|
| `id` | unique score request id |
| `row_id` | original datapool row id |
| `q_tag` | `q1`, `q2`, or eval-specific tag |
| `src` | English source text |
| `mt` | candidate translation |
| `backend` | QE backend name |

Recommended fields:

| Field | Description |
|---|---|
| `run_id` | unique experiment run id |
| `subset_idx` | subset index |
| `phase` | calling phase, such as `infer-q1`, `infer-q2`, or `eval-ood` |

Optional field:

| Field | Description |
|---|---|
| `ref` | reference translation, only for reference-based fallback/eval |

Rules:

- input must be valid JSONL
- one line must be one independent scoring request for one candidate translation
- the subprocess must not accept combined `mt_q1`/`mt_q2` rows
- malformed lines must fail loudly
- reference-free QE must not require `ref`

---

## 11. JSONL Output Contract

QE subprocess output must be JSONL.

Each line represents one backend score.

Example:

```jsonl
{"id":"run_abc123/subsets/subset_000/sample_000001/q1","score":3.41,"backend":"metricx24","model_name":"google/metricx-24-hybrid-xxl-v2p6-bfloat16"}
{"id":"run_abc123/subsets/subset_000/sample_000001/q2","score":5.82,"backend":"metricx24","model_name":"google/metricx-24-hybrid-xxl-v2p6-bfloat16"}
```

Required fields:

| Field | Description |
|---|---|
| `id` | request id from input |
| `score` | raw backend score |
| `backend` | QE backend name |
| `model_name` | QE model name |

Optional fields:

| Field | Description |
|---|---|
| `runtime_ms` | inference latency |
| `error` | error message for failed row |
| `status` | `ok` or `failed` |

Rules:

- stdout must contain JSONL only
- stderr is reserved for logs
- the driver must not output progress bars to stdout
- raw backend score must not be renamed to `qe_q1` or `qe_q2` inside the isolation layer

---

## 12. MetricX Driver Responsibilities

The MetricX subprocess driver must:

1. read JSONL input
2. load tokenizer
3. load MetricX model
4. format each row for reference-free QE
5. tokenize with configured `max_input_length`
6. run batched inference
7. write raw MetricX scores as JSONL

Reference-free input format:

```txt
source: {src} candidate: {mt}
```

Optional reference-based format:

```txt
source: {src} candidate: {mt} reference: {ref}
```

Rules:

- return raw MetricX backend scores
- do not compute Q1/Q2 semantics
- do not convert MetricX error to quality score here unless explicitly assigned to the scoring layer
- do not compute collapse terms
- do not compute `S_i`
- do not select samples
- for OOD reference-based eval, pass `Source_En` as source, the model translation as hypothesis, and `Target_Ko` as reference

OOD eval may also compute BLEU and chrF in the same isolated QE/eval runtime. These lexical metrics do not require GPU model code, but keeping them in the subprocess avoids adding evaluation-only dependencies such as `sacrebleu` to the main training environment.

---

## 13. COMET Driver Responsibilities

The COMET subprocess driver must:

1. read JSONL input
2. load COMET model
3. construct COMET input rows
4. call `model.predict`
5. extract per-sample scores
6. write raw COMET scores as JSONL

Reference-free COMET-Kiwi input:

```json
{
  "src": "English source text",
  "mt": "Korean translation"
}
```

Rules:

- return raw COMET backend scores
- do not compute Q1/Q2 semantics
- do not compute collapse terms
- do not compute `S_i`
- do not select samples
- do not mutate training state

---

## 14. Subset Artifact Boundaries

The final subset artifact layout is owned by the pipeline contract.

Expected subset layout:

```txt
artifacts/
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

QE isolation may write temporary runtime files under the current subset directory.

Recommended temporary layout:

```txt
artifacts/
  runs/
    {run_id}/
      subsets/
        subset_000/
          qe_tmp/
            q1_request.jsonl
            q1_response.jsonl
            q2_request.jsonl
            q2_response.jsonl
            stderr.log
```

Rules:

- QE isolation does not own `scored.jsonl`
- QE scoring owns `scored.jsonl`
- selection owns `selected.jsonl`
- temporary files must be reproducible or safely disposable

---

## 15. Environment Safety

The subprocess runner should set safe defaults.

Recommended override:

```bash
HF_HUB_ENABLE_HF_TRANSFER=0
```

Reason:

Some environments enable `HF_HUB_ENABLE_HF_TRANSFER=1`, but the QE venv may not have `hf_transfer` installed.

The subprocess should inherit useful variables:

```txt
HF_TOKEN
HUGGINGFACE_HUB_TOKEN
CUDA_VISIBLE_DEVICES
```

The subprocess should not depend on training-specific internal state.

---

## 16. Failure Handling

If the subprocess exits with a nonzero return code, the main process must treat QE as failed.

Failure report must include:

- backend
- model name
- return code
- stderr
- stdout prefix
- request path
- response path, if created
- number of input rows
- run id
- subset index
- phase
- run id
- config hash

Example:

```txt
QE subprocess failed
backend=metricx24
model=google/metricx-24-hybrid-xxl-v2p6-bfloat16
returncode=1
run_id=run_abc123
subset_idx=0
phase=score
stderr=...
stdout_prefix=...
```

Rules:

- do not silently continue
- do not fabricate QE scores in production
- do not retry indefinitely
- every failed row or batch must be traceable

---

## 17. Missing Runtime Policy

Recommended config:

```yaml
qe_isolation:
  missing_runtime:
    policy: error
```

Allowed values:

| Policy | Meaning |
|---|---|
| `error` | fail loudly |
| `dummy` | return dummy scores, PoC only |

Default:

```yaml
policy: error
```

Dummy policy rules:

- allowed only for explicit PoC or unit tests
- must log that QE is disabled
- must mark output rows as dummy
- must never be used for production selection

---

## 18. Timeout Handling

Recommended config:

```yaml
qe_isolation:
  subprocess:
    timeout_seconds: 1800
```

Rules:

- timeout is per subprocess call
- timeout failure must be logged
- timed-out batches must not produce trusted partial scores unless explicitly supported
- timeout value must be config-driven

---

## 19. Config Contract

Recommended config:

```yaml
qe_isolation:
  enabled: true

  setup:
    managed_install: false
    runtime: metricx      # metricx | comet | both | skip
    shared_venv_allowed: true
    cuda_wheel_detection: nvidia_smi

  env:
    comet_python_env: COMET_PYTHON
    metricx_python_env: METRICX_PYTHON
    inherit:
      - HF_TOKEN
      - HUGGINGFACE_HUB_TOKEN
      - CUDA_VISIBLE_DEVICES
      - HF_HUB_ENABLE_HF_TRANSFER

  subprocess:
    required: true
    input_format: jsonl
    output_format: jsonl
    capture_output: true
    text: true
    check: false
    timeout_seconds: 1800
    stdout_jsonl_only: true

  missing_runtime:
    policy: error

  safety:
    disable_hf_transfer: true

qe_runtime:
  metricx24:
    batch_size: 8
    max_input_length: 1536
    device: cuda

  comet_kiwi:
    batch_size: 8
    gpus: 1
```

Rules:

- all values must come from YAML
- missing config must fail validation
- no backend parameter may be hardcoded in source
- local paths must be provided through environment variables, not committed config
- `managed_install: false` means the runner expects the QE venv to already exist
- setup scripts may use `runtime: metricx | comet | both | skip`, but production scoring should not silently skip QE
- if `shared_venv_allowed: true`, `COMET_PYTHON` and `METRICX_PYTHON` may point to the same Python binary
- `HF_HUB_ENABLE_HF_TRANSFER` should be forced to `0` when the QE venv lacks `hf_transfer`

---

## 20. Relationship to Other Documents

`docs/qe-isolation.md` defines:

- virtual environment separation
- subprocess execution
- runtime discovery
- JSONL driver contracts
- backend runtime parameters
- failure isolation

`docs/qe-scoring.md` defines:

- Q1 / Q2 semantics
- QE score direction conversion
- collapse definition
- weighted `S_i`
- selection policy

`docs/training.md` defines:

- collapse LoRA lifecycle
- base update lifecycle
- LoRA unload rules

`docs/config-schema.md` defines:

- validation rules for all config sections

Boundary rule:

```txt
QE isolation returns raw backend scores.
QE scoring converts and interprets scores.
Training consumes selected rows only.
```

---

## 21. Implementation Notes for Agents

Preferred layout:

```txt
src/scp_stage4/qe/
  subprocess.py
  runtime.py
  config.py
  drivers/
    metricx_driver.py
    comet_driver.py
```

Implementation order:

1. implement QE runtime discovery
2. implement JSONL reader/writer helpers
3. implement subprocess call wrapper
4. implement MetricX driver
5. implement COMET driver
6. add timeout handling
7. add missing-runtime policy
8. add temporary artifact paths under the subset directory
9. add tests using mocked subprocess responses

The main process must not directly import:

```python
from comet import download_model, load_from_checkpoint
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
```

unless it is inside a QE subprocess driver.

---

## 22. Agent Notes Template

Every implementation task touching QE isolation should end with:

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

## 23. Open Questions

1. Should production use one shared QE venv or split COMET and MetricX venvs after dependency testing?
2. Should timeout be per batch or per full subset scoring job?
3. Should subprocess drivers be checked-in CLI files or generated temporary scripts?
4. Should dummy runtime be allowed in CI tests only?
5. Should temporary `qe_tmp/` files be retained by default?

Resolved: MetricX raw score clamping belongs in the scoring layer. QE isolation must return raw backend scores.

Current defaults:

- subprocess isolation required
- JSONL I/O required
- missing runtime policy: `error`
- one shared `~/.venvs/comet` QE venv is acceptable for PoC
- split backend venvs are acceptable for production
- runtime parameters stored under `qe_runtime`
- final score interpretation happens in `docs/qe-scoring.md`
