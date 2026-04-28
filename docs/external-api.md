# SCP Stage 4 External API

> **Project**: `scp_stage4_sft`  
> **Scope**: External LLM correction calls, provider routing, request/response JSONL schemas, status handling, retry policy, cost logging, and secret safety.

---

## 1. Goal

External LLMs are used to create corrected Korean training labels only for selected fragile samples.

The external API layer must:

- call external LLMs only after sample selection
- send the English source and model draft translation
- receive a corrected full Korean translation or filter status
- log cost, tokens, latency, provider, model, and prompt version
- never log secrets
- write reproducible JSONL artifacts

The external API layer must not:

- select samples
- train models
- run QE scoring
- mutate base model state
- call APIs for unselected monolingual rows

---

## 2. Execution Boundary

External correction happens after `selected.jsonl`.

```txt
scored.jsonl
  ↓
selected.jsonl
  ↓
unload collapse LoRA
  ↓
external API correction
  ↓
api_requests.jsonl + api.jsonl
  ↓
base update training
```

Rules:

- unselected rows must not be sent to external APIs
- eval and OOD rows must not be sent to external APIs during SCP training
- collapse LoRA must already be unloaded before external API correction begins
- API outputs must be written before base update begins
- base update must consume only accepted correction rows

---

## 3. Provider Config

Baseline config:

```yaml
external_api:
  routing:
    mode: single       # single | fallback | ensemble
    fallback_enabled: false
    ensemble_enabled: false

  primary:
    provider: openai
    model: gpt-5.4  # placeholder; replace with an available provider model before real calls
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
```

Rules:

- provider names and model names are config-owned
- `gpt-5.4` is a placeholder model string in this design doc, not a guarantee that the model exists
- real API execution must validate that the configured provider/model is available
- default routing is `single`
- fallback and ensemble schemas may exist but are disabled by default
- API keys are referenced by environment variable name only
- secrets must never appear in config artifacts, logs, prompts, or errors

---

## 4. Request JSONL Schema

The API request artifact should be JSONL.

Recommended file:

```txt
artifacts/
  runs/
    {run_id}/
      subsets/
        subset_000/
          api_requests.jsonl
```

Each line:

```json
{
  "id": "sample_000001",
  "request_id": "run_abc123/subsets/subset_000/sample_000001/api",
  "run_id": "run_abc123",
  "subset_idx": 0,
  "row_id": "sample_000001",
  "dataset": "alwaysgood/reuter_processed",
  "source": "English source text",
  "student": "Korean model draft from mt_q1",
  "metadata": {
    "title": "optional headline",
    "document_type": "article",
    "text_role": "body",
    "original_id": "optional source id"
  },
  "selection": {
    "score_s": 1.27,
    "qe_q1": 0.82,
    "qe_q2": 0.71,
    "delta_qe": -0.11,
    "collapse_term": 0.134
  },
  "prompt_version": "teacher_correction_v1",
  "prompt_hash": "string",
  "provider": "openai",
  "model": "configured-provider-model",
  "status": "ok",
  "config_hash": "sha256-of-effective-config"
}
```

Required fields:

| Field | Meaning |
|---|---|
| `id` | stable row id copied from selected row |
| `request_id` | unique API request id |
| `run_id` | experiment run id |
| `subset_idx` | SCP subset index |
| `row_id` | original selected row id |
| `dataset` | dataset name for traceability |
| `source` | English source text |
| `student` | model draft translation, usually `mt_q1` |
| `metadata` | context such as dataset, document type, text role, and title/headline |
| `selection` | score snapshot (`score_s`, `qe_q1`, `qe_q2`, `delta_qe`, optional `collapse_term`) |
| `prompt_version` | teacher prompt version |
| `prompt_hash` | teacher prompt hash |
| `provider` | configured provider |
| `model` | configured model |
| `status` | request staging status (default `ok`) |
| `config_hash` | run config hash for reproducibility |

Rules:

- `student` should come from `mt_q1`
- `mt_q2` is not sent to the teacher correction prompt
- metadata should be sent to the teacher when available because document type and text role can affect translation style
- metadata is context only and must not become part of the corrected translation unless the source itself is that title/body text
- selection scores may be included for traceability but must not be required by the prompt
- request rows must be reproducible from `selected.jsonl`, prompt config, and provider config

---

## 5. Teacher Output Contract

The teacher prompt returns plain text.

Expected format:

```txt
Line 1: one of [no_change, minor_edit, major_edit, rewrite, invalid]
Line 2+: corrected translation only
```

If line 1 is `invalid`, line 2+ contains a short Korean reason instead of a translation.

Teacher labels:

| Label | Meaning |
|---|---|
| `no_change` | student draft is already correct |
| `minor_edit` | small terminology, grammar, or style fix |
| `major_edit` | substantial correction |
| `rewrite` | draft is mostly wrong; full rewrite |
| `invalid` | source/sample should not be used for training |

The corrected translation must be a full Korean translation of the source, not a summary.

Details are defined in:

```txt
docs/prompts.md
```

---

## 6. Response JSONL Schema

Recommended file:

```txt
artifacts/
  runs/
    {run_id}/
      subsets/
        subset_000/
          api.jsonl
```

Each line:

```json
{
  "id": "sample_000001",
  "request_id": "run_abc123/subsets/subset_000/sample_000001/api",
  "run_id": "run_abc123",
  "subset_idx": 0,
  "row_id": "sample_000001",
  "dataset": "alwaysgood/reuter_processed",
  "metadata": {
    "title": "optional headline",
    "document_type": "article",
    "text_role": "body",
    "original_id": "optional source id"
  },
  "provider": "openai",
  "model": "configured-provider-model",
  "status": "ok",
  "teacher_label": "minor_edit",
  "source": "English source text",
  "student": "Korean model draft from mt_q1",
  "gold": "Corrected full Korean translation",
  "reason": null,
  "prompt_version": "teacher_correction_v1",
  "prompt_hash": "string",
  "usage": {
    "input_tokens": 1200,
    "output_tokens": 360,
    "total_tokens": 1560
  },
  "cost": {
    "currency": "USD",
    "estimated": 0.0
  },
  "latency_ms": 2300,
  "attempt": 1,
  "error": null,
  "config_hash": "sha256-of-effective-config"
}
```

Allowed statuses:

| Status | Meaning | Use for Training |
|---|---|---|
| `ok` | valid correction received | yes |
| `skipped` | intentionally skipped by policy | no |
| `filtered` | teacher judged sample invalid | no |
| `needs_review` | output parsed but validation is uncertain | no by default |
| `failed` | API/runtime failure | no |

Teacher label to status mapping:

| Teacher Label | Status |
|---|---|
| `no_change` | `ok` |
| `minor_edit` | `ok` |
| `major_edit` | `ok` |
| `rewrite` | `ok` |
| `invalid` | `filtered` |

Rules:

- `gold` is required when `status: ok`
- `reason` is required when `status` is `filtered`, `skipped`, `needs_review`, or `failed`
- failed rows must include a sanitized error message
- API responses must preserve request ids
- API responses should preserve `id`, `row_id`, `dataset`, and `metadata` from request rows
- response artifacts must not contain secrets

---

## 7. Retry and Failure Policy

Baseline config:

```yaml
external_api:
  failure:
    retry: 2
    backoff_seconds: 2.0
    backoff_multiplier: 2.0
    on_error: skip
```

Rules:

- retry transient provider errors
- do not retry deterministic validation failures indefinitely
- after retry exhaustion, write `status: failed`
- `failed` rows must not be used for training
- partial API failure must not crash the whole run unless failure rate exceeds configured limits
- every failed attempt must be logged with provider, model, sanitized error, and latency

Baseline config:

```yaml
external_api:
  failure:
    max_failure_rate: 0.2
    on_failure_rate_exceeded: fail
```

---

## 8. Throughput and Concurrency

External API calls can be a major bottleneck.

Baseline config:

```yaml
external_api:
  concurrency:
    max_requests: 8
    max_requests_per_provider: 8
    rate_limit_policy: provider_default
```

Rules:

- concurrency must be config-owned
- retries must preserve deterministic request ids
- response order may differ from request order
- artifacts must be restorable by `row_id`
- provider rate limits must be respected
- cost and latency must be logged per request

Do not improve throughput by skipping logs or merging multiple selected samples into one untraceable prompt.

Artifact rule:

- `api_requests.jsonl` stores rendered, sanitized request records before provider execution
- `api.jsonl` stores one final response/status row per request after retry handling
- both files must share `id`, `request_id`, and `row_id`
- `api.jsonl` is the artifact consumed by base update training

---

## 9. Cost and Usage Logging

Every API response must log:

```yaml
run_id
subset_idx
row_id
request_id
provider
model
prompt_version
prompt_hash
status
teacher_label
input_tokens
output_tokens
total_tokens
estimated_cost
latency_ms
attempt
config_hash
```

Recommended run-level artifact:

```txt
artifacts/
  runs/
    {run_id}/
      api_costs.jsonl
```

Rules:

- cost may be estimated if provider billing finalization is delayed
- cost calculation method must be logged or config-owned
- token usage must come from provider response when available
- if provider usage is unavailable, estimate and mark as estimated

---

## 10. Validation

External outputs must be validated before training.

Validation checks:

- teacher label is allowed
- corrected translation is non-empty for `ok`
- invalid/filter reason is non-empty for `filtered`
- output is not visibly truncated
- output does not include prompt text, markdown wrappers, or explanations
- key numbers and named entities are plausibly preserved

Automatic `needs_review` triggers:

- teacher label is valid but corrected text fails a soft language/format heuristic
- important number, ticker, percentage, or currency preservation is uncertain
- corrected text is suspiciously short or long relative to source and student draft
- output contains mixed commentary and translation but can still be parsed
- metadata/source mismatch is suspected, such as title-style output for a body row
- validator cannot confidently choose between `ok` and `filtered`

Validation outcome:

| Condition | Status |
|---|---|
| valid teacher correction | `ok` |
| teacher returns `invalid` | `filtered` |
| source skipped by local policy | `skipped` |
| parse succeeds but checks are uncertain | `needs_review` |
| API call or parsing fails after retries | `failed` |

Rows with `status != ok` must not be used for base update training unless explicitly configured.

---

## 11. Fallback and Ensemble

Fallback and ensemble are schema-supported but disabled by default.

```yaml
external_api:
  routing:
    mode: single
    fallback_enabled: false
    ensemble_enabled: false
```

Fallback mode may later try a secondary provider after provider failure.

Ensemble mode may later call multiple providers and apply arbitration.

Rules:

- fallback must preserve the original request id and record every provider attempt
- ensemble must store all candidate outputs
- arbitration rules must be config-owned
- default SCP SFT uses `single`

---

## 12. Secret Safety

Secrets must never be committed or logged.

Allowed:

```yaml
api_key_env: OPENAI_API_KEY
```

Forbidden:

```yaml
api_key: sk-...
```

Rules:

- read API keys from environment variables
- redact provider errors before logging if they may contain request headers
- do not include full raw provider responses if they may contain secrets
- never write secrets to `effective_config.yaml`
- never include secrets in prompt text

---

## 13. Relationship to Other Documents

`docs/external-api.md` defines:

- external provider routing
- API request/response schemas
- statuses and retry behavior
- cost and usage logging

`docs/prompts.md` defines:

- teacher correction prompt
- teacher edit labels
- prompt versioning

`docs/logging.md` defines:

- run-level and subset-level logs
- event schemas
- cost aggregation

---

## Agent Notes

- Default routing is OpenAI/single-provider in the example config; example model strings are placeholders until replaced with available provider models.
- Default routing is `single`; fallback and ensemble are schema-only for now.
- Teacher correction receives `source` and `mt_q1` draft, not `mt_q2`.
- `invalid` is a teacher judgment and maps to `filtered`; `failed` is reserved for runtime/API failure.

## Suggested Improvements

- Define provider-specific request adapters when implementation begins.
- Decide whether `needs_review` can be manually promoted to training data.
- Add exact cost formula once provider pricing config is finalized.
