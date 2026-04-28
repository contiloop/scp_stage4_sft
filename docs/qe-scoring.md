# SCP Stage 4 QE Scoring

> Project: `scp_stage4_sft`  
> Scope: QE scoring, MetricX/COMET backend handling, self-collapse measurement, weighted S computation, and sample selection.

---

## 1. Goal

This document defines how SCP Stage 4 computes QE scores and quantifies **self-collapse**.

The QE scoring module must:

- score baseline translations
- score post-probe translations
- normalize QE direction so that **higher always means better**
- compute collapse signals
- compute weighted \(S_i\)
- optionally apply hard filters
- output scored and selected JSONL artifacts

This module must not:

- train models
- generate translations
- call external correction APIs
- mutate training state

---

## 2. Q1 / Q2 Definition

SCP Stage 4 uses two QE scores per source sample.

### Q1: before / baseline QE

`Q1` is the QE score of the model's **high-temperature self-translation** before probe LoRA training.

### Q2: after / probe QE

`Q2` is the QE score of the model's **greedy regeneration** after training a probe LoRA adapter on its own generated output.

In PoC terminology:

```text
delta_qe = qe_after - qe_before
```

So:

```text
delta_qe = Q2 - Q1
```

Collapse corresponds to:

```text
delta_qe < 0
```

Meaning:

```text
Q2 < Q1
```

---

## 3. QE Score Direction Contract

All downstream SCP logic assumes:

```text
higher QE score = better translation quality
```

Therefore every QE backend must expose quality-oriented scores.

### COMET-Kiwi

COMET-Kiwi scores are already quality-oriented.

```text
range: approximately 0 to 1
higher = better
```

Supported models:

```yaml
- Unbabel/wmt23-cometkiwi-da-xl
- Unbabel/wmt22-cometkiwi-da
```

### MetricX-24

MetricX-24 outputs MQM-style error scores.

```text
range: 0 to 25
lower = better
```

Therefore MetricX raw scores must be converted before SCP scoring:

```text
quality = 25 - error
```

After conversion:

```text
higher = better
```

This is not just normalization. It is a required direction conversion from error score to quality score.

Clamp policy:

- QE drivers return raw MetricX error scores unchanged.
- If raw error is NaN or non-finite, mark the row as a QE row failure.
- If raw error is outside `[0, 25]`, preserve the raw value for debugging, clamp only for quality conversion, and set `metricx_clamped: true`.
- Quality conversion uses `quality = 25 - clamp(raw_error, 0, 25)`.
- Clamp and direction conversion belong to the scoring layer, not the isolated runtime driver.

Supported model:

```yaml
- google/metricx-24-hybrid-xxl-v2p6-bfloat16
```

MetricX-24 hybrid models support both reference-free QE and reference-based evaluation.

For OOD evaluation with references, use:

```json
{
  "source": "Source_En text",
  "hypothesis": "model Korean translation",
  "reference": "Target_Ko reference"
}
```

MetricX raw scores remain MQM-style error scores where lower is better. SCP scoring must convert them to quality-oriented scores when storing fields such as `qe_q1` or `qe_q2`. OOD eval may additionally store raw reference-based MetricX error scores for analysis.

OOD reference metrics:

```yaml
ood_eval:
  neural_metric: metricx24_ref
  overlap_metrics:
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

- `metricx24_ref` runs in reference-based mode using `Source_En`, model hypothesis, and `Target_Ko`
- store MetricX reference output as both raw error and converted quality when useful
- use `ood/metricx24_ref_quality_mean` for best-checkpoint comparison
- BLEU and chrF are reporting metrics and are higher-is-better
- BLEU and chrF should be computed in the QE/eval subprocess runtime so evaluation dependencies stay isolated from the main training runtime
- BLEU must match the reference notebook: `sacrebleu.metrics.BLEU(effective_order=True, smooth_method="exp")`
- chrF must match the reference notebook: `sacrebleu.metrics.CHRF(word_order=2)`
- both BLEU and chrF are computed with row-level `sentence_score(hypothesis, [reference]).score`; aggregate means are logged to W&B

---

## 4. QE Backend Config

Example:

```yaml
qe:
  primary:
    backend: metricx24   # metricx24 | comet_kiwi
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
```

COMET example:

```yaml
qe:
  primary:
    backend: comet_kiwi
    model_name: Unbabel/wmt23-cometkiwi-da-xl
    fallback_model_name: Unbabel/wmt22-cometkiwi-da
    batch_size: 8
    gpus: 1
    score_direction: higher_is_better
    transform:
      type: none
```

Rules:

- all runtime QE configuration lives under `qe.*`
- `qe_scoring.*` is deprecated PoC terminology and must not be used in implementation config

---

## 5. COMET Usage Reference

COMET-Kiwi reference-free QE uses `src` and `mt`.

```python
from comet import download_model, load_from_checkpoint

model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
model = load_from_checkpoint(model_path)

data = [
    {
        "src": "The output signal provides constant sync so the display never glitches.",
        "mt": "Das Ausgangssignal bietet eine konstante Synchronisation, so dass die Anzeige nie stört."
    }
]

model_output = model.predict(data, batch_size=8, gpus=1)
```

The implementation must extract per-sample scores from `model_output`.

---

## 6. Environment Isolation Requirement

QE must run in a separate Python environment.

Reason:

- training stack may require a different `transformers` version
- COMET and MetricX may require incompatible dependency versions
- importing QE models inside the training process can break the run

The main SCP process must call QE through a subprocess.

Expected environment variables:

```bash
COMET_PYTHON=/path/to/comet-venv/bin/python
METRICX_PYTHON=/path/to/metricx-venv/bin/python
```

If `METRICX_PYTHON` is not set, the implementation may reuse `COMET_PYTHON` only if that environment supports MetricX dependencies.

Details belong in:

```text
docs/qe-isolation.md
```

---

## 7. Input / Output Contract

### QE Subprocess Input JSONL

The isolated QE runtime scores one translation per row.

```json
{
  "id": "run_abc123/subsets/subset_000/sample_000001/q1",
  "row_id": "sample_000001",
  "q_tag": "q1",
  "backend": "metricx24",
  "src": "string",
  "mt": "string",
  "run_id": "run_abc123",
  "subset_idx": 0,
  "phase": "infer-q1"
}
```

Rules:

- one row = one `(row_id, q_tag, mt)` scoring request
- `q_tag` must be `q1`, `q2`, or an eval-specific tag such as `ood`
- the scoring layer calls the isolated QE subprocess twice for SCP probing, once for Q1 rows and once for Q2 rows, unless the implementation batches both tags into one request file with the same one-translation-per-row schema
- the QE subprocess must not receive `mt_q1` and `mt_q2` in the same row

### Scoring Layer Joined Input

After QE subprocess results are returned, the scoring layer joins Q1 and Q2 by `row_id`:

```json
{
  "id": "sample_000001",
  "row_id": "sample_000001",
  "source": "string",
  "mt_q1": "string",
  "qe_q1": 24.18,
  "qe_raw_q1": 0.82,
  "metricx_q1_clamped": false,
  "mt_q2": "string",
  "qe_q2": 24.07,
  "qe_raw_q2": 0.93,
  "metricx_q2_clamped": false
}
```

### Output JSONL

```json
{
  "id": "sample_000001",
  "qe_q1": 24.18,
  "qe_raw_q1": 0.82,
  "metricx_q1_clamped": false,
  "qe_q2": 24.07,
  "qe_raw_q2": 0.93,
  "metricx_q2_clamped": false,
  "delta_qe": -0.11,
  "collapse_term": 0.134,
  "difficulty_term": -3.185523,
  "difficulty_z": 0.42,
  "collapse_z": 1.11,
  "score_s": 0.91
}
```

Rules:

- `qe_q1` and `qe_q2` must be quality-oriented
- `qe_raw_q1` and `qe_raw_q2` preserve raw backend score direction
- MetricX rows should record `metricx_q1_clamped` / `metricx_q2_clamped` when clamp-on-convert is enabled
- `delta_qe = qe_q2 - qe_q1`
- collapse corresponds to `delta_qe < 0`
- MetricX raw error scores must not be written as `qe_q1` or `qe_q2`
- selected rows should additionally include `selection_rank` and `selection_rule`

---

## 8. Weighted S Score

Current experimental score:

\[
S_i^{(w)} = \alpha \cdot Z\!\left(-\log(Q_{1,i} + \epsilon)\right) + \beta \cdot Z\!\left(c_i(Q_{1,i}, Q_{2,i})\right),
\qquad \beta = 1 - \alpha
\]

with:

\[
c_i(Q_{1,i}, Q_{2,i}) = \max\!\left(\frac{Q_{1,i} - Q_{2,i}}{Q_{1,i} + \epsilon},\; 0\right)
\]

Default:

```yaml
weighted_score:
  enabled: true
  alpha: 0.3
  beta: 0.7
```

Where:

- `Q1_i`: before-QE quality score
- `Q2_i`: after-QE quality score
- `epsilon`: small constant
- `Z`: z-score standardization
- `c_i(Q1_i, Q2_i)`: collapse term

Interpretation:

- `-log(Q1 + epsilon)` captures baseline difficulty
- `c_i` captures self-collapse and is computed from both `Q1_i` and `Q2_i`
- high `S_i` means difficult and collapse-prone

---

## 9. Collapse Term

Current default:

\[
c1_i = c_i(Q_{1,i}, Q_{2,i}) = \max\!\left(\frac{Q_{1,i} - Q_{2,i}}{Q_{1,i} + \epsilon},\; 0\right)
\]

Equivalent interpretation:

```text
if Q2 < Q1:
    collapse exists
else:
    collapse = 0
```

Config:

```yaml
collapse_term:
  type: c1
  collapse_rate_threshold: 0.0
```

The collapse term must be modular.

Future variants may replace `c1`, including log-ratio or squared variants from PoC experiments.

`qe/collapse_rate` is defined as the fraction of scored rows where `c_i > collapse_rate_threshold`. With the default threshold, this is equivalent to the fraction of rows where `Q2 < Q1`.

---

## 10. Standardization

Before combining terms, apply z-score separately:

```text
difficulty_z = Z(-log(Q1 + epsilon))
collapse_z = Z(c_i)
```

Default:

```yaml
standardization:
  method: zscore
  scope: batch
```

Future option:

```yaml
scope: dataset
```

---

## 11. Hard Filter

Hard filters may be added before computing `S_i`.

Example:

```yaml
hard_filter:
  enabled: false
  rules: []
```

Possible future filters:

```yaml
hard_filter:
  enabled: true
  rules:
    - name: min_q1
      field: qe_q1
      op: ">="
      value: 0.05
    - name: max_metricx_error
      field: metricx_raw_error
      op: "<="
      value: 25.0
```

Rules:

- hard filters run before weighted scoring
- every filtered row must be logged
- filters must be config-driven
- no hidden filtering in code

---

## 12. Selection Policy

Selection uses `score_s`, but the exact selector is experiment-owned.

The selection config should support composable conditions rather than a single hardcoded formula.

Baseline example:

```yaml
selection:
  policy: configurable
  default_rule:
    require_score_s_gte: 0.0
    top_fraction: 0.10
  rules: []
```

Possible future examples:

```yaml
selection:
  policy: threshold
  min_score_s: 0.05

selection:
  policy: top_fraction
  top_fraction: 0.10

selection:
  policy: expression
  expression: "score_s >= 0 and rank_pct <= 0.10"
```

Rules:

- deterministic
- config-driven
- logged
- independent of training state
- every selected row must record which selection rule accepted it
- every non-selected row should remain reproducible from `scored.jsonl`

---

## 13. Failure Handling

```yaml
failure:
  retry: 2
  on_error: skip
```

If QE scoring fails:

1. retry
2. if still failing, skip the sample
3. log sample id, backend, model name, and error message

Fallback dummy scorers may be used only for PoC dry-runs, not production SCP.

---

## 14. Artifacts

Recommended output:

```text
artifacts/
  qe/
    scored.jsonl
    selected.jsonl
    failures.jsonl
```

Meanings:

- `scored.jsonl`: all successfully scored rows
- `selected.jsonl`: rows selected for training
- `failures.jsonl`: failed or filtered rows

---

## 15. Full Config Example

```yaml
qe:
  epsilon: 1e-6

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

    hard_filter:
      enabled: false
      rules: []

    selection:
      policy: configurable
      default_rule:
        require_score_s_gte: 0.0
        top_fraction: 0.10
      rules: []

  failure:
    retry: 2
    on_error: skip

  subprocess:
    required: true
    comet_python_env: COMET_PYTHON
    metricx_python_env: METRICX_PYTHON
```

---

## 16. Implementation Notes for Agents

Preferred module layout:

```text
src/scp_stage4/qe/
  config.py
  backends.py
  metricx.py
  comet.py
  subprocess.py
  scoring.py
  collapse.py
  selection.py
  io.py
```

Implementation order:

1. implement backend score direction normalization
2. implement MetricX subprocess scorer
3. implement COMET subprocess scorer
4. compute `qe_q1`, `qe_q2`
5. compute `delta_qe`
6. compute collapse term
7. standardize terms
8. compute weighted `S_i`
9. apply selection
10. write artifacts

Do not import:

- training modules
- LoRA modules
- external API modules

---

## 17. Open Questions

1. Should MetricX or COMET be the default backend?
2. Should z-score scope be batch-level or dataset-level?
3. Should `c1` be replaced by log-ratio collapse?
4. Should hard filters remove very low Q1 samples before scoring?
5. Should MetricX raw errors be stored for debugging?

Current defaults:

- primary backend: MetricX-24
- MetricX raw error converted with `quality = 25 - error`
- Q1/Q2 stored as quality-oriented scores
- `delta_qe = Q2 - Q1`
- collapse if `delta_qe < 0`
- collapse term: `c1`
- alpha = 0.3
- beta = 0.7
- z-score scope: batch
