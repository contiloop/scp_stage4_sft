# SCP Stage 4 Logging

> **Project**: `scp_stage4_sft`  
> **Default Tracking Project**: `scp_main`  
> **Scope**: Local JSONL logs, artifacts, W&B metrics, Weave traces, API cost tracking, failure logs, and research trajectory analysis.

---

## 1. Goal

SCP Stage 4 is a research pipeline, not just a training script.

Logging must support:

- debugging failed runs
- reproducing every subset
- analyzing selected sample trajectories
- measuring API cost and latency
- tracking training and QE metrics
- preparing future replay buffer or preference tuning data

Hard rule:

```txt
If a decision affects data, model weights, selection, cost, or evaluation, it must be visible in logs or artifacts.
```

---

## 2. Required Context Fields

Every log event must include:

```yaml
run_id
subset_idx
phase
config_hash
```

Recommended additional fields:

```yaml
timestamp
event_type
row_id
request_id
provider
model
prompt_version
prompt_hash
artifact_path
status
error
```

Rules:

- `run_id` identifies one experiment run
- `subset_idx` identifies the SCP iteration
- `phase` identifies the pipeline step
- `config_hash` links the event to the saved effective config
- secrets must never appear in logs

---

## 3. Local Artifact Layout

Recommended layout:

```txt
artifacts/
  runs/
    {run_id}/
      effective_config.yaml
      config_hash.txt
      events.jsonl
      metrics.jsonl
      failures.jsonl
      api_costs.jsonl
      selected_index.jsonl

      subsets/
        subset_000/
          input.jsonl
          q1.jsonl
          q2.jsonl
          scored.jsonl
          selected.jsonl
          api_requests.jsonl
          api.jsonl
          events.jsonl
          metrics.jsonl
          failures.jsonl
          train_final/

      archives/
        subsets/
          subset_000.tar.gz
          subset_000.manifest.json
```

Run-level logs are for cross-subset analysis.

Subset-level logs are for local debugging and reproducibility.

Rules:

- subset-level events may be duplicated or summarized into run-level logs
- JSONL is preferred for append-only logs
- every artifact path should be relative to the run root when possible
- failed/skipped/filtered rows must be logged, not silently dropped
- when subset archive is enabled, archive creation must be logged with `phase: archive-subset`
- subset archive manifest should include run/subset ids, config hash, archive path, and file inventory
- if subset directories are pruned after archive, keep an `ARCHIVED.json` pointer artifact in each pruned subset directory

---

## 4. Logging Config

Baseline config:

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
    events_jsonl: events.jsonl
    metrics_jsonl: metrics.jsonl
    failures_jsonl: failures.jsonl
    api_costs_jsonl: api_costs.jsonl
    selected_index_jsonl: selected_index.jsonl

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
- W&B is used for experiment metrics and artifacts
- Weave is used for LLM/application traces
- all logging behavior must be config-owned
- `redact_inputs` may be enabled if source text cannot be stored externally

---

## 5. W&B Metrics

W&B should track run-level and subset-level metrics.

Recommended metrics:

```yaml
subset/input_rows
subset/q1_rows
subset/q2_rows
subset/scored_rows
subset/selected_rows
subset/api_ok_rows
subset/api_filtered_rows
subset/api_failed_rows
subset/train_rows

qe/q1_mean
qe/q2_mean
qe/delta_mean
qe/collapse_rate   # fraction of scored rows with collapse_term > configured threshold
qe/score_s_mean
qe/score_s_p95

api/input_tokens
api/output_tokens
api/total_tokens
api/estimated_cost
api/latency_ms_mean
api/latency_ms_p95

train/loss
train/learning_rate
train/grad_norm
train/num_steps
checkpoint/saved
checkpoint/retained_count
checkpoint/deleted_count
ood/metricx24_ref_raw_error_mean
ood/metricx24_ref_quality_mean
ood/metricx24_ref_quality_delta_from_previous
ood/bleu_mean
ood/chrf_mean
```

Rules:

- log subset-level metrics with `subset_idx`
- log trainer/internal metrics with trainer `global_step` and also include `subset_idx`
- log OOD eval metrics with `subset_idx`; do not use trainer `global_step` as the primary OOD x-axis
- metric names should be stable across runs
- aggregate metrics must be reproducible from JSONL artifacts
- W&B summaries should not be the only source of truth
- OOD MetricX-24, BLEU, and chrF should be logged together after every subset completion
- W&B charts should compare `ood/metricx24_ref_quality_mean`, `ood/bleu_mean`, and `ood/chrf_mean` on the same `subset_idx` axis

---

## 6. Weave Tracing

Weave should trace external API correction calls and selected application-level functions.

Recommended traced operations:

- `external_api.correct_one`
- `external_api.correct_batch`
- `external_api.parse_teacher_response`
- `external_api.validate_correction`
- `qe.score_batch` if safe and useful

For OpenAI calls, Weave can automatically trace supported SDK calls after `weave.init("scp_main")`. The correction wrapper should also be decorated or instrumented so traces include SCP metadata.

Recommended trace metadata:

```json
{
  "run_id": "run_abc123",
  "subset_idx": 0,
  "row_id": "sample_000001",
  "request_id": "run_abc123/subsets/subset_000/sample_000001/api",
  "phase": "call-api",
  "provider": "openai",
  "model": "configured-provider-model",
  "prompt_version": "teacher_correction_v1",
  "prompt_hash": "string",
  "config_hash": "string"
}
```

Rules:

- trace one selected row as one logical correction call
- preserve `request_id` in both local JSONL and Weave trace metadata
- do not rely on Weave as the only artifact store
- local JSONL must remain complete even if Weave is disabled or unavailable
- provider errors in traces must be sanitized

---

## 7. External API Trace Policy

External API tracing is especially important because correction quality affects training labels.

Each API correction trace should capture:

- rendered prompt or prompt hash according to redaction config
- source text unless redacted
- student draft unless redacted
- teacher raw output
- parsed teacher label
- final `gold` or filter reason
- token usage
- estimated cost
- latency
- retry attempt
- validation result

If `logging.weave.redact_inputs: true`, traces should store:

- row id
- prompt hash
- source hash
- student hash
- metadata
- status
- usage/cost/latency

and local artifacts should remain the complete source of truth.

---

## 8. Event JSONL Schema

Example event:

```json
{
  "timestamp": "2026-04-28T12:00:00Z",
  "run_id": "run_abc123",
  "subset_idx": 0,
  "phase": "score",
  "event_type": "phase_completed",
  "config_hash": "string",
  "metrics": {
    "scored_rows": 32,
    "selected_rows": 8
  },
  "artifact_path": "subsets/subset_000/scored.jsonl",
  "status": "ok",
  "error": null
}
```

Required fields:

| Field | Meaning |
|---|---|
| `timestamp` | event time |
| `run_id` | run identifier |
| `subset_idx` | subset iteration |
| `phase` | pipeline phase |
| `event_type` | event category |
| `config_hash` | effective config hash |
| `status` | `ok`, `skipped`, `filtered`, `needs_review`, `failed` |

---

## 9. Failure Logs

Every failure must be structured.

Example:

```json
{
  "timestamp": "2026-04-28T12:00:00Z",
  "run_id": "run_abc123",
  "subset_idx": 0,
  "phase": "call-api",
  "row_id": "sample_000001",
  "request_id": "run_abc123/subsets/subset_000/sample_000001/api",
  "failure_type": "api_timeout",
  "status": "failed",
  "provider": "openai",
  "model": "configured-provider-model",
  "attempt": 3,
  "error": "sanitized timeout message",
  "config_hash": "string"
}
```

Rules:

- failure logs must be JSONL
- errors must be sanitized
- failed row ids must remain traceable
- training failures should fail the run
- API row failures may be skipped according to config
- QE backend failures should fail the run because selection cannot be trusted

---

## 10. Selection Trajectory Logs

SCP's selected samples are a research artifact.

Recommended `selected_index.jsonl` row:

```json
{
  "run_id": "run_abc123",
  "subset_idx": 0,
  "row_id": "sample_000001",
  "dataset": "alwaysgood/reuter_processed",
  "source_tokens": 512,
  "qe_q1": 0.82,
  "qe_q2": 0.71,
  "delta_qe": -0.11,
  "score_s": 1.27,
  "selection_rank": 3,
  "api_status": "ok",
  "teacher_label": "minor_edit",
  "used_for_training": true
}
```

Use cases:

- inspect which samples became fragile over time
- analyze domain/topic movement across subsets
- debug selection collapse
- build future replay buffers
- build future preference datasets from `student` vs `gold`

---

## 11. Artifact Logging to W&B

Recommended W&B artifacts:

- effective config
- config hash
- data artifacts
- scored and selected JSONL
- API correction JSONL
- train checkpoints or checkpoint metadata
- eval summaries

Rules:

- large checkpoints may be disabled with `log_model_checkpoints: false`
- local artifacts remain authoritative
- W&B artifact names should include `run_id` and optionally `subset_idx`
- do not upload secrets
- checkpoint retention decisions must be logged locally
- OOD eval summaries should be logged to W&B and local metrics JSONL
- OOD eval should include MetricX-24 reference-based score, BLEU, and chrF when references are available

---

## 12. Relationship to Other Documents

`docs/logging.md` defines:

- local JSONL logs
- W&B metrics/artifacts
- Weave tracing policy
- failure and cost logs

`docs/external-api.md` defines:

- API request/response schema
- provider routing
- API status semantics

`docs/config-schema.md` defines:

- config ownership
- Hydra composition
- logging config placement

---

## Agent Notes

- W&B and Weave default project is `scp_main`.
- Weave is best used for external API correction traces and optional wrapper-level application traces.
- Local JSONL artifacts remain the source of truth even when W&B/Weave are enabled.
- Trace redaction is config-owned because source/news text may require privacy control.

## Suggested Improvements

- Add exact implementation examples after the logging module exists.
- Decide whether to upload full source/student/gold text to Weave by default in production.
- Add provider-specific cost calculation config once pricing is finalized.
