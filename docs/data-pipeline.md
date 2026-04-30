# docs/data-pipeline.md

# SCP Stage 4 Data Pipeline

> **Project**: `scp_stage4_sft`  
> **Scope**: Datapool loading, row normalization, train/eval split, OOD test loading, length handling, multiprocessing, and JSONL artifact generation.

---

## 1. Goal

The data pipeline prepares English source rows for SCP Stage 4.

It must:

1. load configured Hugging Face datasets
2. extract English text from known source columns
3. preserve useful document metadata such as title/headline, document type, and text role
4. split datapool into train/eval
5. optionally load an external OOD test set
6. handle long rows safely using tokenizer-aware length logic
7. write deterministic JSONL artifacts for downstream SCP steps

The data pipeline must **not** perform:

- translation inference
- QE scoring
- LoRA training
- external API calls
- wandb/weave orchestration

---

## 2. Data Sources

Initial Hugging Face datasets:

```yaml
data:
  datasets:
    - name: alwaysgood/reuter_processed
      split: train
    - name: alwaysgood/c4_noblocklist_processed
      split: train
    - name: alwaysgood/Bloomberg_Financial_News_processed
      split: train
    - name: alwaysgood/earnings_call_mono
      split: train
    - name: alwaysgood/sec-10k-pre-processed
      split: train
```

Important repository-layout rule:

- If a Hub dataset repo contains non-training JSON artifacts (for example `*_checkpoint.json` or stats files) next to training shards, always set `data.datasets[*].data_files` explicitly to the train shard pattern.
- Do not rely on implicit repository-wide file discovery for mixed-layout repos, because it can introduce schema-cast failures and expensive fallback paths.

Pre-optimization dataset structure check:

- Before implementing throughput optimizations, inspect the dataset repository tree and README YAML config to confirm the true train file layout.
- Verify that `data_files` patterns target only train shards and exclude operational metadata artifacts.
- Run a small-sample dry load (for example with a low `max_rows_per_dataset`) to validate resolved file count, schema consistency, and row-id assumptions before full-scale runs.

Expected English source columns:

```yaml
data:
  text_columns:
    - source_text
    - text
```

Optional article metadata columns:

```yaml
data:
  title_columns:
    - metadata.Headline
    - metadata.headline
    - metadata.title
    - Headline
    - headline
    - title
```

Translatable field policy:

```yaml
data:
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
```

Rules:

- article titles/headlines may be translated as their own rows
- article body text and title/headline should not be concatenated by default
- each generated row must preserve `text_role` so prompts and analysis can distinguish title vs body
- duplicate title/body rows must have distinct deterministic ids

---

## 3. Row Unit

Default unit is **one dataset row**.

Do not split rows into sentences by default.

Long rows may be split only when they exceed tokenizer-aware length limits. See §8.

---

## 4. Base Output Schema

Each normalized datapool row must use JSONL and follow this schema:

```json
{
  "id": "string",
  "dataset": "string",
  "source": "string",
  "metadata": {
    "title": "string|null",
    "document_type": "article|filing|earnings_call|other|null",
    "text_role": "title|body|section|other",
    "original_id": "string|null",
    "parent_id": "string|null",
    "chunk_idx": "int|null"
  }
}
```

Rules:

- `source` is the English text used for translation.
- `dataset` must preserve the original dataset name.
- `metadata.title` stores headline/title if available.
- `metadata.document_type` stores the source genre when known, such as article, filing, or earnings call.
- `metadata.text_role` identifies whether `source` is a title/headline, body text, section heading, or other text unit.
- `parent_id` and `chunk_idx` are only used when a long row is split.
- The data module must not create downstream fields such as `mt_q1`, `qe_q1`, `gold`, etc.

Downstream handoff requirement:

- row ids produced here become the stable ids for `q1`, `q2`, `scored`, `selected`, `api_requests`, `api`, and `train_final` artifacts
- later subset steps may fail if row-id drift is detected
- `clean_base.json` is not produced by prepare-data, but API/update-base phases must be blocked until clean-base verification passes in subset runtime

---

## 5. Text Extraction

For each raw row:

1. Prefer `source_text` if present and non-empty.
2. Else use `text` if present and non-empty.
3. Else skip the row.
4. Extract title/headline if available.
5. Normalize whitespace.

If `data.translatable_fields` is configured, emit one normalized row per non-empty configured field. For article data, this means the title/headline can become a separate translation target with `metadata.text_role: title`, while the article body remains a separate row with `metadata.text_role: body`.

Pseudo-logic:

```python
source = row.get("source_text") or row.get("text")

title = (
    row.get("metadata", {}).get("Headline")
    or row.get("metadata", {}).get("headline")
    or row.get("metadata", {}).get("title")
    or row.get("Headline")
    or row.get("headline")
    or row.get("title")
)
```

Title row pseudo-logic:

```python
if title:
    emit_row(
        source=title,
        metadata={
            "title": title,
            "text_role": "title",
            "document_type": infer_document_type(dataset, row),
        },
    )
```

Body row pseudo-logic:

```python
if source:
    emit_row(
        source=source,
        metadata={
            "title": title,
            "text_role": "body",
            "document_type": infer_document_type(dataset, row),
        },
    )
```

Metadata should help downstream prompts without changing the source text itself.

---

## 6. Minimal Filtering Policy

The upstream datasets are assumed to be mostly preprocessed.

Therefore, avoid aggressive filtering.

Allowed default filtering:

- skip missing source
- skip empty source after whitespace normalization
- handle over-length rows according to §8

Do **not** add additional quality filters unless explicitly configured.

---

## 7. Train / Eval Split

The main datapool is split into:

```txt
train → SCP loop
eval  → in-distribution validation / tracking only
```

Default eval ratio:

```yaml
data:
  split:
    eval_ratio: 0.02
    seed: 42
```

Rules:

- `eval_ratio` must be config-driven.
- Default value is `0.02`.
- Split must be deterministic given `seed`.
- Eval rows must not be used for SCP training or external API correction.
- Eval rows are for tracking translation quality only.

---

## 8. Length Handling

Length handling must be tokenizer-aware.

Do not rely only on character count.

### 8.1 Why Tokenizer-Aware Length Matters

Training/inference prompts include instruction/template tokens, source tokens, and generated output tokens.

Therefore:

```txt
total_tokens = prompt_tokens + source_tokens + generated_output_tokens
```

If `source_tokens` alone is allowed up to `model.max_length`, generation may overflow or truncate the output.

`max_new_tokens` is output-only. It is not the combined prompt + source + output length.

The data pipeline must reserve output budget when deciding whether a source row is safe for inference and training.

### 8.2 Config-Driven Length Policy

All length values must come from YAML config.

The numbers below are a current baseline example, not hardcoded constants. If model context, VRAM, batching policy, or target output length changes, update the YAML config first. Implementation must read the effective config and must not bake these values into source code.

```yaml
model:
  max_length: 8192

data:
  length:
    enabled: true
    mode: tokenizer

    max_total_tokens: 8192
    max_source_tokens: 4000
    max_output_tokens: 4096
    min_available_output_tokens: 768

    safety_margin_tokens: 64
    overflow: split
    output_budget_strategy: dynamic

    split:
      unit: sentence
      max_source_tokens_per_chunk: 4000
      max_chunks_per_row: 4
      min_chunk_tokens: 32
      fallback_for_long_sentence: skip

    generation:
      on_truncation: discard
```

### 8.3 Source Budget Rule

The effective source budget should be computed as:

```txt
available_output_budget =
  model.max_length
  - prompt_template_tokens
  - source_tokens
  - safety_margin_tokens
```

Then:

```txt
effective_max_new_tokens =
  min(data.length.max_output_tokens, inference.*.max_new_tokens, available_output_budget)
```

The row is eligible only if:

```txt
source_tokens <= data.length.max_source_tokens
available_output_budget >= data.length.min_available_output_tokens
```

`min_available_output_tokens` is not a minimum generation length. It means the row must leave at least this many tokens of possible generation space after prompt and source tokens are counted. If less space remains, the row should be split or skipped before inference.

Example:

```txt
model.max_length = 8192
prompt_template_tokens = 120
source_tokens = 4000
safety_margin_tokens = 64

available_output_budget = 8192 - 120 - 4000 - 64 = 4008
effective_max_new_tokens = min(4096, 4096, 4008) = 4008
```

The baseline `max_source_tokens: 4000` preserves more financial/news context while still requiring at least `min_available_output_tokens: 768` of possible output room under an `8192` token total budget.

If a row needs the full `max_output_tokens: 4096`, its source budget is lower:

```txt
source_budget_for_full_output = 8192 - 120 - 4096 - 64 = 3912
```

Therefore the default policy is dynamic:

- shorter sources may use up to the configured `max_output_tokens`
- longer sources may use less output room
- rows with insufficient output room are split or skipped before inference
- generated outputs that hit the effective limit are discarded

There is no hard requirement that token limits be powers of two. Values such as `8192` and `4096` are convenient baseline examples because they are easy to reason about, align with common model context settings, and make padding/bucketing policies simpler. They are still config values, not code constants.

### 8.4 Output Truncation Policy

Generated output must be checked after inference.

Recommended default:

```yaml
generation:
  on_truncation: discard
```

Rules:

- if generation stops because it reached `max_new_tokens`, discard the row
- do not train on visibly truncated Korean outputs
- write discarded rows to a failure or filtered artifact
- include the reason, configured limit, actual generated token count, and row id
- do not silently truncate targets for training

The same policy applies to Q1, Q2, and external-correction outputs when truncation can be detected.

### 8.5 Overflow Policy

Supported policies:

```yaml
overflow: split     # recommended default
overflow: skip
overflow: truncate
```

Recommended default:

```yaml
overflow: split
```

Policy behavior:

| Policy | Behavior | Recommendation |
|---|---|---|
| `split` | Split long row into multiple shorter rows | Default |
| `skip` | Drop over-length row | Safe fallback |
| `truncate` | Hard truncate text | Avoid unless explicitly required |

### 8.6 Sentence-Aware Split

When a row exceeds `max_source_tokens`, split it by sentence boundaries into multiple chunks.

Each chunk becomes a new row.

Example:

```json
{
  "id": "alwaysgood/reuter_processed:123__chunk_0",
  "dataset": "alwaysgood/reuter_processed",
  "source": "First chunk text...",
  "metadata": {
    "title": "Example headline",
    "original_id": "123",
    "parent_id": "alwaysgood/reuter_processed:123",
    "chunk_idx": 0
  }
}
```

Rules:

- preserve `parent_id`
- preserve title/headline metadata
- preserve dataset name
- keep chunks deterministic
- do not create more than `max_chunks_per_row`
- if splitting would exceed `max_chunks_per_row`, mark the original row as skipped with reason `max_chunks_exceeded` unless config explicitly chooses `error`
- if a single sentence exceeds the budget, use `fallback_for_long_sentence`

Default fallback:

```yaml
fallback_for_long_sentence: skip
on_max_chunks_exceeded: skip
```

Hard truncation should be a last resort.

---

## 9. Throughput-Oriented Data Design

The pipeline must be designed for high throughput.

SCP can be slow because it includes generation, temporary LoRA training, QE scoring, external API correction, and base updates. The data layer should reduce wasted tokens and avoid creating pathological batches.

Recommended design:

- pre-tokenize source rows once and cache token counts
- store prompt token counts per prompt version
- compute source/output budgets before inference
- bucket rows by token length
- batch by token budget, not only by row count
- keep deterministic row ids even when rows are bucketed or reordered for throughput
- write enough metadata to restore original datapool order when needed
- split long rows before GPU inference
- discard rows whose generated outputs are truncated

Recommended config:

```yaml
data:
  throughput:
    enabled: true
    pretokenize: true
    cache_token_counts: true
    bucketing:
      enabled: true
      boundaries: [256, 512, 1024, 1536, 2048]
      shuffle_within_bucket: true
    batching:
      strategy: token_budget
      max_batch_tokens: 32768
      pad_to_multiple_of: 8
```

Rules:

- bucketing must not change sample identity
- shuffling must be deterministic given the configured seed
- `max_batch_tokens` must include prompt + source + expected output budget
- padding should be minimized for long rows
- rows above the configured budget should be split or skipped before reaching inference
- all throughput behavior must be controlled by YAML

Notes:

- Powers of two are not required for individual row limits.
- Padding to a small multiple such as `8` can help tensor-core efficiency.
- Token-budget batching is usually more important than choosing power-of-two row limits.

---

## 10. OOD Test Set

A separate OOD test set may be provided as CSV.

Default expected file:

```txt
data/test.csv
```

Expected columns:

```txt
Source_En
Target_Ko
```

Config:

```yaml
data:
  ood_test:
    enabled: true
    path: data/test.csv
    source_column: Source_En
    target_column: Target_Ko
```

Purpose:

- track OOD English→Korean translation quality
- compare stage-to-stage improvement
- evaluate against human/reference Korean target
- provide `Target_Ko` references for OOD metrics such as MetricX-24 reference-based scoring, BLEU, and chrF

Rules:

- OOD test rows must never be used for training.
- OOD test rows must never be sent to external API for correction during SCP training.
- OOD test may later be migrated to a Hugging Face dataset, but the schema must remain explicit.
- `Target_Ko` is the reference column for OOD evaluation.

---

## 11. Multiprocessing

Data processing must support multiprocessing.

Default:

```yaml
data:
  runtime:
    prepare_data:
      intermediate_format: parquet
      parquet_row_group_size: 4096
      progress_enabled: true
      progress_every_rows: 100000
      progress_every_seconds: 10.0
    hf:
      dataset_download_workers: 2
  num_workers: 4
  length:
    tokenizer_batch_size: 16384
```

Rules:

- `num_workers` must come from config.
- Default is `4`.
- `runtime.hf.dataset_download_workers` controls parallelism across datasets.
- `num_workers` controls Hugging Face `load_dataset(..., num_proc=...)` when `streaming=false`.
- `length.tokenizer_batch_size` controls batch length counting when `length.mode=tokenizer`.
- `runtime.prepare_data.intermediate_format=parquet` stores normalized intermediates as Parquet before final JSONL emission.
- `runtime.prepare_data.parquet_row_group_size` controls Parquet row-group flush size.
- `runtime.prepare_data.progress_*` controls periodic rows/s progress logs printed during normalize/split phases.
- When `pyarrow` is unavailable, `prepare-data` automatically falls back to JSONL intermediate mode.
- Use multiprocessing for dataset loading/downloading; normalization and split writing must remain deterministic.
- Output must remain deterministic.
- Random operations must use configured seed.

---

## 11.1 Streaming Architecture (Current Prepare-Data)

`prepare-data` now runs in a streaming architecture to avoid loading full datapool lists in memory.

Execution shape:

```txt
raw row iterator
  → normalized row iterator
  → length policy iterator (batched tokenizer length counting)
  → write datapool.normalized.parquet (streaming intermediate)
  → deterministic train/eval split from intermediate artifact (second pass)
  → write datapool.normalized.jsonl
  → write train/eval/sample artifacts
```

Details:

- `raw_rows -> normalized -> filtered` list materialization is removed.
- Rows are validated and written to `datapool.normalized.parquet` as they stream.
- Final JSONL artifacts are emitted from the intermediate stream in the split pass.
- `tokenizer.encode` per-row loops are replaced with batch length counting (`tokenizer_batch_size`).
- Train/eval split count remains deterministic (`ceil(total * eval_ratio)` with seed).
- Sampling (`first_n` or `random`) is applied while building train/eval outputs, without loading full train rows into memory.

Operational visibility (`prepare_data_summary.json`):

```json
{
  "normalized_rows": 1032456,
  "train_rows": 1011807,
  "eval_rows": 20649,
  "sampled_rows": 1011807,
  "length_policy": {
    "input_rows": 1098723,
    "output_rows": 1032456,
    "split_input_rows": 42117,
    "split_output_rows": 29104,
    "skipped_overflow_policy": 0,
    "skipped_truncate_budget": 0,
    "skipped_long_sentence": 1834,
    "skipped_max_chunks_exceeded": 6427,
    "skipped_split_empty": 2906,
    "skipped_total": 11167
  }
}
```

Interpretation guide:

- `input_rows - output_rows` should approximately match `skipped_total` after accounting for split amplification.
- A rising `skipped_max_chunks_exceeded` usually means `max_chunks_per_row` is too strict for the current corpus.
- A rising `skipped_long_sentence` usually means many rows contain single very long sentences and `fallback_for_long_sentence` may need adjustment.

---

## 12. Subset Construction

SCP subsets are the unit of adaptive training.

Baseline config:

```yaml
pipeline:
  subset:
    strategy: fraction
    fraction: 0.02
    min_size: 32
    max_size: null
    shuffle: true
    seed: 42
    drop_last: false
```

Meaning:

```txt
subset_size = ceil(train_datapool_size * fraction)
```

With `fraction: 0.02`, a datapool of 10,000 rows produces subsets of about 200 rows, or roughly 50 subsets per full pass.

Rules:

- processing order is `load raw rows → normalize/emitted translatable fields → length split/filter → write normalized artifact → deterministic train/eval split → subset construction`
- subset construction must be deterministic given seed and effective config
- every row must preserve its original `row_id`
- subset membership must be reproducible without relying on runtime state
- `subset_idx` is the SCP iteration index
- shuffling and bucketing must not destroy traceability
- eval and OOD rows must never be included in SCP subsets

---

## 13. Sampling for Dry Runs

Support small subsets for fast local development.

Config:

```yaml
data:
  subset_size: 32
  sampling:
    strategy: random
    seed: 42
```

Rules:

- `subset_size` applies only to train datapool unless explicitly configured otherwise.
- Sampling must happen after normalization and split.
- Dry-run subset should not alter the underlying train/eval artifact.
- dry-run `subset_size` overrides normal subset fraction only for development runs

Supported initial strategies:

- `random`
- `first_n`

Future strategies:

- `per_dataset_balanced`
- `weighted_by_dataset`

---

## 14. Dataset Mixing

When multiple datasets are loaded, preserve source dataset identity.

Each row must include:

```json
{
  "dataset": "alwaysgood/reuter_processed"
}
```

Optional future config:

```yaml
data:
  dataset_weights:
    alwaysgood/reuter_processed: 1.0
    alwaysgood/c4_noblocklist_processed: 1.0
    alwaysgood/Bloomberg_Financial_News_processed: 1.0
    alwaysgood/earnings_call_mono: 1.0
    alwaysgood/sec-10k-pre-processed: 1.0
```

Initial implementation does not need weighted sampling unless required.

---

## 15. Artifacts

Recommended artifact layout:

```txt
artifacts/
  data/
    datapool.normalized.parquet
    datapool.normalized.jsonl
    datapool.train.jsonl
    datapool.eval.jsonl
    datapool.train.sampled.jsonl
    ood_test.jsonl
```

Meanings:

| File | Purpose |
|---|---|
| `datapool.normalized.parquet` | normalized intermediate store used by split pass |
| `datapool.normalized.jsonl` | full normalized datapool |
| `datapool.train.jsonl` | train split for SCP loop |
| `datapool.eval.jsonl` | in-distribution eval split |
| `datapool.train.sampled.jsonl` | optional dry-run subset |
| `ood_test.jsonl` | external OOD evaluation set |

Subset runtime artifacts are produced under run roots:

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
          clean_base.json
          collapse_adapter/
          train_final/
```

Post-completion archive policy:

- keep stepwise artifacts explicit while a subset/stage is still running
- after completion, optional subset-level archive can bundle one subset directory into one archive file
- optional deletion of original subset directories is allowed only after the stage is complete
- if original subset directory is removed, keep a pointer file such as `ARCHIVED.json` in its place

---

## 16. Config Contract

All behavior must be controlled through YAML.

Example:

```yaml
model:
  max_length: 8192

data:
  datasets:
    - name: alwaysgood/reuter_processed
      split: train
    - name: alwaysgood/c4_noblocklist_processed
      split: train
    - name: alwaysgood/Bloomberg_Financial_News_processed
      split: train
    - name: alwaysgood/earnings_call_mono
      split: train
    - name: alwaysgood/sec-10k-pre-processed
      split: train

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

  title_columns:
    - metadata.Headline
    - metadata.headline
    - metadata.title
    - Headline
    - headline
    - title

  split:
    eval_ratio: 0.02
    seed: 42

  length:
    enabled: true
    mode: tokenizer
    tokenizer_batch_size: 16384
    max_total_tokens: 8192
    max_source_tokens: 4000
    max_output_tokens: 4096
    min_available_output_tokens: 768
    safety_margin_tokens: 64
    overflow: split
    output_budget_strategy: dynamic
    split:
      unit: sentence
      max_source_tokens_per_chunk: 4000
      max_chunks_per_row: 4
      min_chunk_tokens: 32
      fallback_for_long_sentence: skip
    generation:
      on_truncation: discard

  throughput:
    enabled: true
    pretokenize: true
    cache_token_counts: true
    bucketing:
      enabled: true
      boundaries: [256, 512, 1024, 1536, 2048]
      shuffle_within_bucket: true
    batching:
      strategy: token_budget
      max_batch_tokens: 32768
      pad_to_multiple_of: 8

  ood_test:
    enabled: true
    path: data/test.csv
    source_column: Source_En
    target_column: Target_Ko

  runtime:
    prepare_data:
      intermediate_format: parquet
      parquet_row_group_size: 4096
      progress_enabled: true
      progress_every_rows: 100000
      progress_every_seconds: 10.0

  subset_size: 32
  seed: 42
  num_workers: 4

  sampling:
    strategy: random

pipeline:
  subset:
    strategy: fraction
    fraction: 0.02
    min_size: 32
    max_size: null
    shuffle: true
    seed: 42
    drop_last: false
```

---

## 17. Implementation Notes for Agents

When implementing this module:

1. Start with config schema.
2. Implement text extraction helpers.
3. Implement row normalization as an iterator.
4. Implement tokenizer-aware length validation with batched token counting.
5. Implement sentence-aware splitting for overflow.
6. Implement deterministic train/eval split.
7. Implement streaming JSONL writers and second-pass split artifact builder.
8. Add multiprocessing after single-process logic is correct.
9. Add unit tests with mocked rows.

Preferred layout:

```txt
src/scp_stage4/data/
  config.py
  extraction.py
  normalization.py
  length.py
  splitting.py
  sampling.py
  datapool.py
  io.py
```

Do not import:

- training modules
- inference modules
- QE modules
- external API modules

---

## 18. Open Questions

Before large-scale runs, decide:

1. Should dataset weights be used?
2. Should semantic deduplication be added later?
3. Should OOD test be loaded from local CSV or Hugging Face?
4. Should over-length single sentences be skipped or truncated?
5. Should student model prompts include metadata by default or only when configured experiments require it?

Current defaults:

- one row = one sample
- title/headline may be emitted as a separate translatable row when `data.translatable_fields` enables it
- title/headline is also preserved as metadata for body rows
- no aggressive filtering
- eval ratio = `0.02`
- overflow = sentence-aware split
- long single sentence fallback = skip

---
