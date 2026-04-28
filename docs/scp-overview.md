# SCP Stage 4 Overview

> **Project**: `scp_stage4_sft`  
> **Scope**: High-level motivation, SCP loop, self-collapse probing, adaptive data construction, and research boundaries.

---

## 1. Motivation

SCP Stage 4 exists to improve English-to-Korean translation with limited parallel data.

Supervised translation training ideally requires large-scale parallel data. In this project, however, the available large-scale training source is primarily English monolingual data. One possible solution is to use external LLMs to translate or correct every monolingual sample into Korean, but this is expensive and inefficient.

Not every source sentence is equally useful for training. The current model may already translate some sentences well enough. Sending those easy samples to an external LLM wastes API budget and may increase overfitting, because the model is repeatedly trained on examples it already handles.

The goal of SCP is therefore:

```txt
Use external LLM correction only where the current model is fragile.
```

SCP is a model-adaptive data construction loop. It probes the current model, selects fragile samples, expands only those samples into parallel data, and updates the model subset by subset.

The final goal is to build an English-to-Korean translation-specialized model that is smaller than the external teacher LLMs used for labeling and does not depend on those external LLMs at inference time. The exact target size is config- and experiment-owned rather than fixed in this overview.

---

## 2. Core Idea

SCP uses self-collapse as a probing signal.

When a model is trained on its own generated outputs, its probability distribution does not degrade uniformly. Fragile, ambiguous, or unstable samples tend to collapse earlier than robust samples.

SCP exploits this behavior:

1. The base model generates an initial Korean translation.
2. That translation is treated as a pseudo-label.
3. A temporary weak LoRA adapter is trained on the pseudo-label.
4. The model regenerates the translation with the collapse adapter.
5. The difference between before-collapse and after-collapse quality is used to detect fragile samples.

The collapse adapter is not meant to improve the model. It is a probe. Its purpose is to reveal weakness.

---

## 3. Q1, Q2, and Fragility

For each subset, SCP computes two translation outputs and two QE scores.

```txt
Q1: quality of the base model translation before collapse probing
Q2: quality of the regenerated translation after temporary collapse LoRA
```

The intended flow is:

```txt
English source
  ↓
Q1 inference with base model
  ↓
train temporary collapse LoRA using Q1 as pseudo-label
  ↓
Q2 inference with collapse LoRA
  ↓
compute collapse-aware score S_i
  ↓
select fragile samples
```

Q1 uses config-defined high-temperature sampling to expose the model's current generative distribution. The baseline configuration uses `temperature: 1.1`.

Q2 uses deterministic or greedy decoding after the collapse adapter is applied.

A fragile sample is not simply a bad sample.

A sample may be important if:

- Q1 is already low, meaning the base model likely translated poorly.
- Q1 looks acceptable, but Q2 drops sharply after collapse probing.
- The sample is domain-specific, ambiguous, or unstable under self-training.

The score `S_i` should capture both baseline difficulty and collapse sensitivity.

---

## 4. Why QE Is Used

The main monolingual datapool does not have reference Korean translations.

Therefore, reference-free QE metrics are used as practical proxies for translation quality. Neural QE metrics such as MetricX or COMET-Kiwi are not perfect, but they are currently the most useful available signal for deciding which model outputs are likely weak.

This limitation matters especially because the target domain includes financial and economic text. General-purpose QE models may not fully capture domain-specific terminology, style, or correctness. SCP therefore treats QE as a selection signal, not as a ground-truth oracle.

The expected role of QE is:

```txt
good enough to guide SFT data selection,
not perfect enough to define final translation preference.
```

Further style and preference optimization may be handled in a later post-tuning stage, such as preference tuning or reinforcement learning. That later stage is outside the implementation scope of this repository.

---

## 5. External LLM Correction

External LLMs are called only after fragile samples have been selected.

The correction input may include:

- English source text
- the model's Korean translation, usually `mt_q1`
- optional metadata such as title/headline, document type, text role, and dataset
- project-specific translation instructions

For article datasets, title/headline can also be a translation target by itself. In that case the title is emitted as a separate row with `metadata.text_role: title`, while the article body is emitted separately with `metadata.text_role: body`.

The external LLM should primarily correct the model's translation rather than generate an unrelated translation from scratch. However, the final output must be a correct Korean translation. If the model output is completely wrong, the external LLM must rewrite it fully.

This correction-style target is preferred because it tends to stay closer to the current model's output distribution than a completely independent translation. Empirically, the strongest improvement is expected when the model learns from labels produced by correcting its own output.

The external LLM may also enforce project-specific translation conventions, such as preserving source expressions in parentheses after proper nouns when configured. It may also reject abnormal source rows when the input is not suitable for training. In that case, the output artifact should record a skipped status instead of silently dropping the row.

Provider-specific details, schemas, retry behavior, and cost logging are defined in:

```txt
docs/external-api.md
```

---

## 6. Adaptive Data Construction

SCP is not a one-pass data generation process.

As the base model improves, the set of selected samples should change. Easy samples should gradually stop being selected, while remaining selected samples should reflect the model's current weaknesses.

This creates an adaptive learning schedule:

```txt
early stage: many basic translation weaknesses
later stage: harder, more domain-specific, or more ambiguous weaknesses
```

The desired outcome is a low-cost, model-adaptive parallel data construction process.

Success can be measured by:

- improved English-to-Korean translation quality
- improved in-domain and OOD evaluation performance
- better performance under the same external API budget
- similar performance with fewer external API calls than full monolingual expansion
- meaningful changes in selected sample characteristics across stages

---

## 7. Subset-Level SCP Loop

SCP operates sequentially over subsets.

```txt
subset_i
  ↓
Q1 inference with current base model
  ↓
train temporary collapse LoRA on Q1 pseudo-labels
  ↓
Q2 inference with collapse LoRA
  ↓
QE scoring and S_i computation
  ↓
select fragile samples
  ↓
external LLM correction
  ↓
unload collapse LoRA
  ↓
update base model using corrected data
  ↓
subset_i+1
```

The base model update after one subset affects all later subsets.

The collapse LoRA does not persist. It is created for one subset, used only for probing, and unloaded immediately after Q2 artifacts are produced. External API correction and base update must run with a clean base model.

---

## 8. Evaluation Strategy

SCP uses two different evaluation signals for two different purposes.

During subset selection, the monolingual datapool has no Korean reference. SCP therefore uses reference-free QE to estimate Q1/Q2 quality and compute fragility.

After each subset update, the model must be evaluated on the held-out OOD test set. This OOD evaluation uses reference translations and is not part of selection or training.

The default OOD evaluation contract is:

```txt
source: Source_En
hypothesis: model Korean translation
reference: Target_Ko
```

Default OOD metrics:

```txt
MetricX-24 reference-based score
BLEU
chrF
```

This separation is important. Reference-free QE is useful for choosing which monolingual samples deserve external correction, but it should not be the only signal used to judge whether the base model is actually improving. OOD reference-based evaluation gives a stable subset-to-subset view of progress, regression, and checkpoint quality.

W&B should log MetricX-24 reference-based quality, BLEU, and chrF on the same `subset_idx` axis so the run can be inspected as a trajectory rather than as disconnected final scores.

---

## 9. Artifacts and Reproducibility

Every intermediate artifact must be explicit and reproducible.

Recommended subset layout:

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

Each subset should be reproducible from:

- config
- seed
- dataset artifact
- base checkpoint
- prompt version
- external LLM correction artifact

The system should avoid hidden state. If a decision affects training, selection, cost, filtering, or evaluation, it should appear in config, logs, or artifacts.

---

## 10. Logging and Research Value

SCP must be heavily logged because the selection trajectory is itself a research artifact.

The system should make it possible to inspect:

- which samples were selected at each subset
- how Q1, Q2, collapse terms, and `S_i` changed over time
- what external LLM corrections were produced
- how much API cost was spent
- which samples were skipped or filtered
- how model updates affected later selection behavior

This is important for debugging, reproducibility, and research analysis.

The logs should also leave room for future extensions. A replay buffer may be added if forgetting becomes a problem. Later preference tuning may use pairs consisting of the model's original output and the external LLM correction, but that post-tuning stage is not implemented in this repository.

Detailed logging contracts are defined in:

```txt
docs/logging.md
```

---

## 11. Implementation Validation Strategy

Implementation and real execution happen in different environments.

Local development should validate logic and contracts quickly without depending on GPU availability, real QE models, or external API calls. Local validation should stay lightweight: Hydra config loading, schema checks, JSONL artifact shape, row-id preservation, mocked QE/API behavior, and tiny fixture smoke tests are enough.

External instance validation should verify the real runtime. That includes GPU model loading, Q1/Q2 generation, collapse LoRA training, isolated MetricX/COMET execution, external API smoke calls, W&B/Weave logging, and a dry-run subset.

This split is intentional:

```txt
local: fast contract validation
remote: real runtime validation
```

The first implementation milestone should therefore be the harness validation foundation:

- Makefile targets
- Hydra config validation
- JSONL schema checker
- local smoke fixtures
- remote validation targets

Only after this foundation exists should the full SCP subset loop be connected.

---

## 12. Boundaries

This repository implements the SCP Stage 4 SFT pipeline.

It does not implement:

- full monolingual-to-parallel expansion
- blind external LLM translation for every sample
- preference tuning or reinforcement learning
- direct QE imports inside the main training runtime
- merging collapse LoRA into the base model

Hard boundaries:

- Collapse LoRA is temporary and must be unloaded after probing.
- Base update is persistent and uses external LLM corrected data.
- External LLM calls happen only after selection.
- QE is a proxy signal, not a ground-truth oracle.
- All intermediate artifacts must be explicit and reproducible.

---

## Agent Notes

- The overview defines conceptual boundaries only. Exact YAML fields belong in `docs/config-schema.md`.
- Q1 uses config-defined higher-temperature sampling; baseline `temperature` is `1.1`.
- External LLM correction is correction-first, but it must produce a correct Korean translation even when full rewriting is required.
- Preference tuning and reinforcement learning are future stages and are not implemented here.
- Local implementation validation does not require real GPU, QE model execution, or external API calls.
- External instance validation is responsible for real runtime checks.

## Suggested Improvements

- Add concrete config examples in `docs/config-schema.md`.
- Define external API output statuses in `docs/external-api.md`, including `ok`, `skipped`, and `failed`.
- Define prompt versions and prompt hashing in `docs/prompts.md`.
- Define selection trajectory logs in `docs/logging.md`.
- Implement the harness validation foundation before connecting the full SCP loop.
