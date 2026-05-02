# SCP Stage 4 Open Questions

> **Project**: `scp_stage4_sft`  
> **Scope**: Remaining design decisions, experiment-dependent defaults, and implementation choices that should not be hidden in code.

---

## 1. Purpose

This document tracks unresolved questions for the SCP Stage 4 harness.

Open questions should be resolved by:

- explicit user decision
- small PoC experiment
- benchmark result
- implementation constraint discovered during build

Do not silently resolve these in code.

---

## 2. Training Backend Details

### Q1. Is Unsloth-patched TRL `SFTTrainer` allowed? — Resolved

Stage 3 `train.py` uses:

- Hydra entrypoint
- Unsloth model loading
- Unsloth LoRA/full-finetuning setup
- TRL `SFTTrainer`
- W&B/Weave helpers

Stage 4 training must use Unsloth.

Decision:

```txt
Unsloth + TRL SFTTrainer is allowed when the model has been loaded,
patched, and prepared through Unsloth.
```

Clarification:

- Plain Hugging Face Trainer or PEFT-only paths that bypass Unsloth are forbidden.
- TRL `SFTTrainer` is acceptable as the training loop for Unsloth-prepared models.
- Stage 3 training patterns may be adapted, but Stage 4 subset lifecycle rules still apply.
- Packing, response template, chat template, EOS behavior, max sequence length, dtype, and padding side must be explicit in training/model config or resolved into the effective config.

---

### Q2. How should cumulative base update checkpoints be represented?

Base update is cumulative across subsets.

Options:

- keep one main adapter and continue training it
- save one checkpoint after each subset
- save both rolling latest and immutable per-subset checkpoints

Current leaning:

```txt
Save after each subset, maintain a latest pointer, keep the latest 2 checkpoints, keep the best 2 checkpoints by OOD metric, and always keep final.
```

Baseline retention is documented in `docs/training.md`.

---

### Q3. When should full-weight update be used?

Config supports:

```yaml
base_update.mode: lora | full_weight
```

Decision needed:

- should full-weight be enabled only for explicit experiments?
- should full-weight require additional validation because it is more expensive and riskier?

Current default:

```txt
LoRA by default, full_weight only when explicitly configured.
```

---

## 3. Data and Length Policy

### Q4. Are the baseline token budgets stable after real VRAM testing?

Current baseline:

```yaml
model.max_length: 8192
data.length.max_source_tokens: 3900
data.length.max_output_tokens: 4096
data.length.min_available_output_tokens: 768
```

Decision needed after testing:

- can Q1/Q2 inference maintain throughput under this budget?
- does collapse LoRA training fit comfortably on the target GPU?
- does QE scoring become the dominant bottleneck?

Current default:

```txt
Use current baseline, but keep all values config-owned.
```

---

### Q5. Should title/headline be prepended to source or kept only as metadata? — Partially Resolved

Decision:

```txt
Title/headline can be a separate translatable row, and it is also preserved as metadata for related body rows.
```

Rules:

- Do not concatenate title and body by default.
- Preserve `metadata.text_role` so title/headline rows can be analyzed separately from body rows.
- Send useful metadata to the external teacher for context.
- Include metadata in the student model prompt only when explicitly configured and consistently used for Q1/Q2, SFT, and eval inference.

Open sub-question:

```txt
Should student prompts include metadata by default in the first experiment?
```

Current default:

```txt
Student metadata inclusion is disabled by default and must be explicitly enabled in config.
```

---

## 4. QE Runtime and Scoring

### Q6. Shared QE venv or split backend venvs in production?

PoC uses one shared venv:

```txt
~/.venvs/comet/bin/python
```

Both `COMET_PYTHON` and `METRICX_PYTHON` may point to it.

Decision needed:

- keep shared venv for production simplicity?
- split COMET and MetricX venvs if dependencies conflict?

Current default:

```txt
Shared venv allowed for PoC; split venvs allowed for production.
```

---

### Q7. Should MetricX remain the primary QE backend?

Current config baseline:

```yaml
qe.primary.backend: metricx24
```

Decision depends on:

- VRAM availability
- runtime speed
- stability
- correlation with human/reference checks

Current default:

```txt
MetricX-24 primary, COMET-Kiwi alternative.
```

For OOD eval, MetricX-24 reference-based scoring is the current default because `Target_Ko` references are available.

---

### Q8. Should MetricX raw score clamping happen in driver or scoring layer? — Resolved

Current docs say:

```txt
QE isolation returns raw backend scores.
QE scoring converts and interprets scores.
```

Decision:

```txt
QE isolation returns raw backend scores. The scoring layer owns interpretation, direction conversion, and clamping.
```

Rules:

- NaN or non-finite raw scores are row-level QE failures.
- Out-of-range MetricX raw errors are preserved for debugging.
- Only the quality conversion path clamps raw errors to `[0, 25]`.
- Clamped rows must set a debug flag such as `metricx_clamped: true`.

---

## 5. External API and Teacher Labels

### Q9. Should `no_change` rows be used for training? — Resolved

Teacher label:

```txt
no_change
```

means the student draft is already correct.

Options:

- train on `no_change` rows because they are validated gold
- skip them to reduce overfitting on already-good examples
- sample them with a lower weight

Decision:

```txt
Train on no_change rows by default because the teacher has validated them as correct gold labels.
```

Config:

```yaml
training:
  base_update:
    data:
      use_no_change: true
```

---

### Q10. Should `needs_review` ever be promoted to training data?

`needs_review` means parsing succeeded but validation was uncertain.

Decision needed:

- never train on `needs_review`
- allow manual review/promotion
- allow automatic promotion if checks pass later

Current default:

```txt
Do not train on needs_review.
```

---

### Q11. What is the exact proper-noun and English-term style guide?

Teacher prompt currently says:

- preserve proper nouns
- use Korean form with original in parentheses when useful
- keep English terms commonly used in Korean finance

Decision needed:

- add examples for company names
- add examples for tickers
- add examples for product names
- define when to use Korean transliteration vs original English

Current default:

```txt
Prompt rule exists, examples still needed.
```

---

### Q12. What API failure rate should fail a subset? — Resolved

Current API policy:

```yaml
retry: 2
on_error: skip
```

Decision needed:

- if 5%, 10%, or 20% of selected rows fail, should the subset fail?
- should this depend on selected row count?

Decision:

```txt
Fail the subset if more than 20% of selected API requests fail after retries.
```

Config:

```yaml
external_api:
  failure:
    max_failure_rate: 0.2
    on_failure_rate_exceeded: fail
```

---

## 6. Logging, W&B, and Weave

### Q13. Should Weave store full source/student/gold text by default?

Current logging config includes:

```yaml
logging.weave.redact_inputs: false
```

Decision needed:

- store full text for easier debugging
- store hashes only for privacy or data-control reasons

Current default:

```txt
Full text allowed by default; can be redacted by config.
```

---

### Q14. Which calls should be traced with Weave?

Recommended:

- external API correction wrapper
- OpenAI SDK call
- parse teacher response
- validate correction

Decision needed:

- should QE subprocess calls also be traced?
- should training subset steps be traced as high-level spans?

Current default:

```txt
Trace external API; do not trace QE subprocess by default.
```

---

### Q15. Should model checkpoints be uploaded to W&B?

Current logging config:

```yaml
logging.wandb.log_model_checkpoints: false
```

Decision needed:

- upload checkpoint metadata only?
- upload LoRA adapters?
- avoid uploads due to size/cost?

Current default:

```txt
Do not upload checkpoints by default.
```

---

## 7. Future Extensions

### Q16. Should a replay buffer be added?

Motivation:

- reduce forgetting
- reuse high-value corrected samples
- maintain difficult examples across subsets

Current default:

```txt
Not implemented in initial Stage 4; keep logs compatible with future replay buffer.
```

---

### Q17. How will preference tuning consume Stage 4 artifacts?

Future stage may use:

```txt
student mt_q1 vs external gold
```

as a preference pair.

Current default:

```txt
Preference tuning is outside this repository, but artifacts should preserve pairs.
```

---

## 8. Stage 3 Code Reuse Candidates

Stage 3 `src/train.py` has useful patterns:

- Hydra `@hydra.main`
- printing resolved config
- W&B env setup
- Weave initialization
- resume checkpoint resolution with `auto`
- bf16/fp16 fallback logic
- trainable parameter counting
- Unsloth backend resolution
- full-weight vs LoRA checks
- metadata column pruning for faster dataloading
- runtime packing options
- trainer metrics saving

Decision needed:

```txt
Which helpers should be copied, adapted, or rewritten for Stage 4?
```

Current default:

```txt
Reuse design patterns, but do not copy Stage 3 training loop blindly.
```

---

## Agent Notes

- This file should shrink over time as decisions are made.
- Resolved decisions should either be moved into the relevant docs or marked as resolved with date/context.
- Do not let open questions become hidden implementation defaults.

## Suggested Improvements

- Add owners or decision dates once active implementation starts.
- Add experiment links for questions resolved by benchmark.
- Create config validation tests for every resolved schema decision.
