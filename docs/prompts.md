# SCP Stage 4 Prompts

> **Project**: `scp_stage4_sft`  
> **Scope**: Translation prompt templates, external teacher correction prompts, prompt versioning, filtering instructions, and output contracts.

---

## 1. Goal

This document defines prompt contracts for SCP Stage 4.

Prompts are part of the experiment configuration. They affect generated translations, external correction labels, selection analysis, and reproducibility.

Rules:

- prompts must be YAML-configured
- prompt versions must be logged
- prompt text must not be hidden in code
- prompt changes must change the effective config hash
- model training prompts and external teacher prompts must be separated

---

## 2. Prompt Types

SCP Stage 4 uses two main prompt families.

| Prompt | Used by | Purpose |
|---|---|---|
| training/inference translation prompt | base model | generate `mt_q1`, `mt_q2`, eval translations |
| external teacher correction prompt | external LLM | produce corrected Korean `gold` labels from source and student draft |

The base model prompt asks for direct English-to-Korean translation.

The external teacher prompt receives the English source and the model's student translation, then returns a corrected full Korean translation or a filter status.

---

## 3. Training / Inference Translation Templates

These templates are used when prompting the trainable model.

Baseline config:

```yaml
prompts:
  translation:
    version: translation_v1
    src_lang_name: English
    src_locale: en-US
    tgt_lang_name: Korean
    tgt_locale: ko-KR
    selection: random
    selection_seed_scope: row_id
    phase_template_policy:
      q1: same_pool
      q2: same_as_q1
      sft: same_as_q1
      eval: deterministic_first
    metadata:
      include_for_student_model: false
      render_format: json
      allowed_fields:
        - document_type
        - text_role
        - title
    templates:
      - "You are a professional {src_lang_name} ({src_locale}) to {tgt_lang_name} ({tgt_locale}) translator. Your goal is to accurately convey the meaning and nuances of the original {src_lang_name} text while adhering to {tgt_lang_name} grammar, vocabulary, and cultural sensitivities. Produce only the {tgt_lang_name} translation, without any additional explanations or commentary. Please translate the following {src_lang_name} text into {tgt_lang_name}: {src}"
      - "Translate the following text from {src_lang_name} to {tgt_lang_name}: {src}"
      - "What does this sentence mean in {tgt_lang_name} from {src_lang_name}: {src}"
      - "How do you translate this sentence into {tgt_lang_name} from {src_lang_name}: {src}"
      - "Translate the following text to {tgt_lang_name}: {src}"
```

Rules:

- output must contain only the Korean translation
- no explanations, comments, labels, or markdown
- template selection must be deterministic given config seed when randomness is used
- when `selection: random`, the default template seed is `(config_seed, row_id)` and does not include `subset_idx`
- `selection_seed_scope: row_id_subset` may include `subset_idx` only for explicit prompt-variation experiments
- Q2 and SFT must reuse the same rendered translation prompt policy as Q1 for the same row unless the experiment explicitly changes this in config
- eval should use a deterministic configured template, not random sampling, unless explicitly configured
- the exact rendered prompt or prompt id must be reproducible from artifacts
- prompt token counts should be cached per template version for throughput
- metadata inclusion for the student model must be config-owned
- if metadata is included during Q1/Q2 inference, the same metadata rendering policy must be used during base-update SFT and eval inference
- do not include metadata in the student prompt unless the metadata will also be available at the intended inference time
- `text_role` may help the model distinguish title/headline translation from body translation when explicitly enabled

---

## 4. External Teacher Correction Prompt

The external teacher prompt is used only after sample selection.

Its purpose is to produce corrected Korean labels for SFT.

Baseline prompt:

```txt
You are a senior translation editor for English-to-Korean financial and economic text.

Review the source and the student translation. The student translation may contain errors. Use it as a draft only. Your job is not to summarize or explain. Your job is to produce a correct, complete Korean translation of the full source text.

Requirements:
- Preserve numbers, dates, currencies, percentages, units, tickers, company names, proper nouns, and financial terminology.
- Use the provided metadata only as context for translation style and disambiguation.
- Do not omit, summarize, simplify, or add information that is not in the source.
- Use natural Korean grammar while preserving the meaning, nuance, and domain-specific terminology of the English source.
- If a proper noun is commonly written in Korean, use the Korean form and include the original expression in parentheses when useful for disambiguation.
- If an English term is commonly used as-is in Korean financial writing, keep the English term.
- If the student translation is already correct, return it unchanged.
- If the student translation is partially wrong, edit it.
- If the student translation is fundamentally wrong, rewrite it fully.
- If the source is invalid, nonsensical, not English, corrupted, unsafe to train on, or impossible to translate reliably, return invalid.

Output format:
Line 1: one of [no_change, minor_edit, major_edit, rewrite, invalid]
Line 2+: the corrected translation only.

If Line 1 is invalid, Line 2+ must contain a short Korean reason instead of a translation.

Source:
{source}

Metadata:
{metadata}

Student translation:
{student}
```

Notes:

- `student` should usually be `mt_q1`.
- The teacher should use `mt_q1` as a draft, but the final label must be a correct full translation of `source`.
- The teacher must not prefer a flawed student translation only because it is close to the model distribution.
- External teacher prompts should receive useful metadata such as `document_type`, `text_role`, dataset name, and title/headline when available.
- Metadata is context only; it must not be translated unless it is also the `source` for that row.
- `{metadata}` is rendered as compact JSON by default, with stable key ordering and secrets excluded.

---

## 5. Teacher Edit Labels

Teacher output line 1 uses an edit label.

| Label | Meaning | Training Use |
|---|---|---|
| `no_change` | student translation is already correct | may train if accepted |
| `minor_edit` | small terminology, grammar, or style correction | train |
| `major_edit` | meaning-level correction required | train |
| `rewrite` | student translation is mostly wrong and must be rewritten | train |
| `invalid` | source/sample should not produce a training target | do not train |

Mapping to external API artifact status:

| Teacher Label | API Status |
|---|---|
| `no_change` | `ok` |
| `minor_edit` | `ok` |
| `major_edit` | `ok` |
| `rewrite` | `ok` |
| `invalid` | `filtered` |

`needs_review` may be assigned by post-processing when the output format is valid but policy checks are uncertain.

`failed` is reserved for API/runtime failure, not for teacher judgment.

---

## 6. Filtering Policy

The teacher may return `invalid` when the source row is unsuitable for training.

Examples:

- source is empty or corrupted
- source is not English
- source is mostly code, tables, boilerplate, or unreadable fragments
- source contains insufficient context to translate reliably
- source is unsafe or inappropriate for the training target
- source is duplicated metadata rather than translatable content

Rules:

- invalid rows must be written to artifacts
- invalid rows must not be silently dropped
- invalid rows must not be used for base update training
- the reason should be logged without exposing secrets

---

## 7. Output Validation

External teacher responses must be parsed and validated.

Validation rules:

- line 1 must be one of the allowed edit labels
- line 2+ must be non-empty
- if line 1 is not `invalid`, line 2+ must be Korean translation text
- output must not include explanations, markdown, JSON wrappers, or repeated source text
- output must not be visibly truncated
- output must preserve key numbers and named entities unless translation convention requires adaptation

Rows that fail validation should receive status `needs_review` or `failed` according to `docs/external-api.md`.

---

## 8. Prompt Versioning

Every prompt family must have a version.

Recommended config:

```yaml
prompts:
  translation:
    version: translation_v1
  teacher_correction:
    version: teacher_correction_v1
```

Required artifact fields:

```json
{
  "prompt_version": "teacher_correction_v1",
  "prompt_hash": "string",
  "template_id": "string|null"
}
```

Rules:

- prompt hash must be computed from the rendered prompt template or canonical prompt config
- changing prompt text must change `prompt_hash`
- prompt version/hash must be included in API logs and correction artifacts
- do not reuse a version name for semantically different prompts

---

## 9. Relationship to Other Documents

`docs/prompts.md` defines:

- model-facing translation prompt templates
- external teacher correction prompt
- teacher edit labels
- prompt versioning

`docs/external-api.md` defines:

- provider routing
- request/response JSONL schema
- retry and failure behavior
- API cost and latency logging

`docs/config-schema.md` defines:

- where prompt config lives
- config ownership and validation

---

## Agent Notes

- The teacher prompt is correction-first, but correctness of the full Korean translation has priority.
- `invalid` is a teacher judgment and maps to `filtered`, not `failed`.
- The teacher prompt does not include `mt_q2`; external correction is based on the source and `mt_q1` draft.

## Suggested Improvements

- Add Korean style examples for proper nouns and financial terms.
- Decide whether `no_change` rows should always be used for training or optionally skipped.
- Add automatic validation rules for number/entity preservation.
