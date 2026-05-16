#!/usr/bin/env bash
# Auto-resuming wrapper around `step_subset run-stage`.
#
# run-stage has no built-in resume: a NCCL hang / OOM just kills it. This
# loop detects how far the run got (per-subset artifact markers), restarts
# run-stage from the first incomplete subset and, within that subset, from
# the first incomplete phase, then keeps going until run-stage exits 0.
#
# Usage:
#   scripts/run_stage_autoresume.sh run_main_001 [extra OVERRIDES...]
#
# Env knobs (optional):
#   CONFIG          default configs/scp_stage4_real.yaml
#   MAX_RETRIES     default 1000
#   COOLDOWN_SECS   default 30
#   LOG_DIR         default artifacts/runs/<run_id>/autoresume_logs
#   USE_FULL_TRAIN_DATA default 1
#   PYTORCH_ALLOC_CONF default expandable_segments:True
#   TORCH_FR_BUFFER_SIZE default 1048576

set -u

RUN_ID="${1:?usage: run_stage_autoresume.sh <run_id> [overrides...]}"
shift || true
EXTRA_OVERRIDES=("$@")

CONFIG="${CONFIG:-configs/scp_stage4_real.yaml}"
MAX_RETRIES="${MAX_RETRIES:-1000}"
COOLDOWN_SECS="${COOLDOWN_SECS:-30}"
RUN_ROOT="artifacts/runs/${RUN_ID}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/autoresume_logs}"
USE_FULL_TRAIN_DATA="${USE_FULL_TRAIN_DATA:-1}"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-1048576}"

PHASES=(infer-q1 train-collapse-lora infer-q2 score unload-collapse-lora call-api update-base)

mkdir -p "$LOG_DIR"

# Completion marker file (relative to subsets/subset_NNN/) for each phase.
phase_marker() {
  case "$1" in
    infer-q1)              echo "q1.jsonl" ;;
    train-collapse-lora)   echo "collapse_adapter/adapter_config.json" ;;
    infer-q2)              echo "q2.jsonl" ;;
    score)                 echo "selected.jsonl" ;;
    unload-collapse-lora)  echo "clean_base.json" ;;
    call-api)              echo "api.jsonl" ;;
    update-base)           echo "train_final/checkpoint_state.json" ;;
  esac
}

json_status_ok() {
  local path="$1"
  python3 - "$path" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
except Exception:
    raise SystemExit(1)

if payload.get("status") != "ok":
    raise SystemExit(1)
for row in payload.get("status_rows") or []:
    if row.get("status") != "ok":
        raise SystemExit(1)
PY
}

phase_complete() {
  local subset_dir="$1"
  local phase="$2"
  local marker
  marker="$(phase_marker "$phase")"

  case "$phase" in
    train-collapse-lora)
      [ -f "$subset_dir/collapse_adapter/adapter_config.json" ] \
        && [ -f "$subset_dir/collapse_adapter/adapter_model.safetensors" ] \
        && [ -s "$subset_dir/collapse_adapter/adapter_model.safetensors" ] \
        && json_status_ok "$subset_dir/collapse_adapter/collapse_state.json"
      ;;
    update-base)
      [ -f "$subset_dir/$marker" ] && json_status_ok "$subset_dir/$marker"
      ;;
    infer-q1|infer-q2|score|call-api)
      [ -s "$subset_dir/$marker" ]
      ;;
    unload-collapse-lora)
      [ -f "$subset_dir/$marker" ]
      ;;
    *)
      [ -f "$subset_dir/$marker" ]
      ;;
  esac
}

print_subset_progress() {
  local subset_idx="$1"
  local subset_dir
  subset_dir="$(printf '%s/subsets/subset_%03d' "$RUN_ROOT" "$subset_idx")"

  if [ ! -d "$subset_dir" ]; then
    echo "[autoresume] subset_${subset_idx}: no artifacts yet; starting full subset"
    return
  fi

  local done=()
  local missing=()
  local ph
  for ph in "${PHASES[@]}"; do
    if phase_complete "$subset_dir" "$ph"; then
      done+=("$ph")
    else
      missing+=("$ph")
    fi
  done

  echo "[autoresume] subset_${subset_idx} completed phases: ${done[*]:-none}"
  echo "[autoresume] subset_${subset_idx} next missing phases: ${missing[*]:-none}"
}

cleanup_gpu() {
  pkill -9 -f scp_stage4 2>/dev/null || true
  pkill -9 -f torchrun 2>/dev/null || true
  pkill -9 -f training_worker 2>/dev/null || true
  pkill -9 -f vllm 2>/dev/null || true
  pkill -9 -f EngineCore 2>/dev/null || true
  sleep 5
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | tr -d ' ' | xargs -r kill -9 2>/dev/null || true
}

# Echo "<subset_idx> <start_phase_or_empty>" for where to resume.
compute_resume_point() {
  local idx=0
  while true; do
    local d
    d="$(printf '%s/subsets/subset_%03d' "$RUN_ROOT" "$idx")"
    if [ ! -d "$d" ]; then
      echo "$idx "
      return
    fi
    # update-base marker present => this subset fully done, move on.
    if phase_complete "$d" update-base; then
      idx=$((idx + 1))
      continue
    fi
    # Partial subset: find first phase whose marker is missing.
    for ph in "${PHASES[@]}"; do
      if ! phase_complete "$d" "$ph"; then
        echo "$idx $ph"
        return
      fi
    done
    # All markers present but no update-base checkpoint_state (shouldn't
    # happen) -> redo update-base.
    echo "$idx update-base"
    return
  done
}

attempt=0
while [ "$attempt" -lt "$MAX_RETRIES" ]; do
  attempt=$((attempt + 1))

  read -r SUBSET_IDX START_PHASE < <(compute_resume_point)
  PHASE_ARG=()
  if [ -n "${START_PHASE:-}" ]; then
    # start_from_phase only applies to the first subset; safe even when q1
    # for that subset is missing only if START_PHASE==infer-q1, so guard:
    if [ "$START_PHASE" != "infer-q1" ]; then
      PHASE_ARG=(--start-from-phase "$START_PHASE")
    fi
  fi

  echo "[autoresume] attempt=$attempt subset_idx=$SUBSET_IDX start_phase=${START_PHASE:-<full>}"
  print_subset_progress "$SUBSET_IDX"
  if [ -n "${START_PHASE:-}" ]; then
    echo "[autoresume] resuming subset_${SUBSET_IDX} from phase '${START_PHASE}'"
  else
    echo "[autoresume] starting subset_${SUBSET_IDX} from the beginning"
  fi
  LOG_PATH="$(printf '%s/run_stage_subset_%03d_attempt_%04d.log' "$LOG_DIR" "$SUBSET_IDX" "$attempt")"
  echo "[autoresume] log=$LOG_PATH"

  CMD=(
    python3 -m scp_stage4.pipeline.step_subset run-stage
    --config "$CONFIG"
    --run-id "$RUN_ID"
    --subset-idx "$SUBSET_IDX"
  )
  if [ "$USE_FULL_TRAIN_DATA" = "1" ]; then
    CMD+=(--use-full-train-data)
  fi
  CMD+=(
    "${PHASE_ARG[@]}"
    "${EXTRA_OVERRIDES[@]}"
  )

  PYTHONPATH=src "${CMD[@]}" 2>&1 | tee "$LOG_PATH"
  rc=${PIPESTATUS[0]}

  if [ "$rc" -eq 0 ]; then
    echo "[autoresume] run-stage finished cleanly (attempt=$attempt)"
    exit 0
  fi

  echo "[autoresume] run-stage exited rc=$rc; cleaning GPU and retrying in ${COOLDOWN_SECS}s"
  cleanup_gpu
  sleep "$COOLDOWN_SECS"
done

echo "[autoresume] exhausted MAX_RETRIES=$MAX_RETRIES" >&2
exit 1
