#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/pipeline_common.sh"

BACKBONE="${BACKBONE:-sd15}"
PROMPT_FILE="${PROMPT_FILE:-${RLEPD_PROMPTS_TXT_DEFAULT}}"
SEEDS="${SEEDS:-0-31}"

case "${BACKBONE}" in
  sd15)
    RUN_DIR="${RUN_DIR:-$(latest_run_dir sd15_k20 || true)}"
    CHECKPOINT="${CHECKPOINT:-}"
    OUTDIR_NAME="${OUTDIR_NAME:-regression_sd15}"
    BATCH="${BATCH:-8}"
    require_dir "${RUN_DIR}"
    run_export_predictor "${RUN_DIR}" "${CHECKPOINT}"
    predictor="$(resolve_export_predictor "${RUN_DIR}")"
    require_file "${predictor}"
    MASTER_PORT=24567 python sample.py \
      --predictor_path "${predictor}" \
      --prompt-file "${PROMPT_FILE}" \
      --seeds "${SEEDS}" \
      --batch "${BATCH}" \
      --outdir "samples/${OUTDIR_NAME}"
    ;;
  sd3-512)
    RUN_DIR="${RUN_DIR:-$(latest_run_dir sd3_512_new || true)}"
    CHECKPOINT="${CHECKPOINT:-}"
    OUTDIR_NAME="${OUTDIR_NAME:-regression_sd3_512}"
    MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-4}"
    require_dir "${RUN_DIR}"
    run_export_predictor "${RUN_DIR}" "${CHECKPOINT}"
    predictor="$(resolve_export_predictor "${RUN_DIR}")"
    require_file "${predictor}"
    python sample_sd3.py \
      --predictor "${predictor}" \
      --prompt-file "${PROMPT_FILE}" \
      --seeds "${SEEDS}" \
      --max-batch-size "${MAX_BATCH_SIZE}" \
      --resolution 512 \
      --outdir "samples/${OUTDIR_NAME}"
    ;;
  sd3-1024)
    RUN_DIR="${RUN_DIR:-$(latest_run_dir sd3_1024_continue || true)}"
    CHECKPOINT="${CHECKPOINT:-}"
    OUTDIR_NAME="${OUTDIR_NAME:-regression_sd3_1024}"
    MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-2}"
    require_dir "${RUN_DIR}"
    run_export_predictor "${RUN_DIR}" "${CHECKPOINT}"
    predictor="$(resolve_export_predictor "${RUN_DIR}")"
    require_file "${predictor}"
    python sample_sd3.py \
      --predictor "${predictor}" \
      --prompt-file "${PROMPT_FILE}" \
      --seeds "${SEEDS}" \
      --max-batch-size "${MAX_BATCH_SIZE}" \
      --resolution 1024 \
      --outdir "samples/${OUTDIR_NAME}"
    ;;
  *)
    echo "Unsupported BACKBONE=${BACKBONE}. Expected one of: sd15, sd3-512, sd3-1024" >&2
    exit 1
    ;;
esac

score_all_metrics "${OUTDIR_NAME}" "${PROMPT_FILE}"
