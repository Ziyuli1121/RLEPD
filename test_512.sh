#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/pipeline_common.sh"

RUN_DIR="${RUN_DIR:-exps/20251219-032038-sd3_512_new}"
PROMPT_FILE="${PROMPT_FILE:-${RLEPD_PROMPTS_TXT_DEFAULT}}"
SEEDS="${SEEDS:-0-999}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-4}"
STEPS="${STEPS:-15000 16000 17000 18000}"

require_dir "${RUN_DIR}"
require_file "${PROMPT_FILE}"

for step in ${STEPS}; do
  run_export_predictor "${RUN_DIR}" "checkpoints/policy-step$(printf '%06d' "${step}").pt"
  predictor="$(resolve_export_predictor "${RUN_DIR}" "${step}")"
  require_file "${predictor}"
  python sample_sd3.py \
    --predictor "${predictor}" \
    --prompt-file "${PROMPT_FILE}" \
    --seeds "${SEEDS}" \
    --max-batch-size "${MAX_BATCH_SIZE}" \
    --outdir "samples/512_${step}"
done

for step in ${STEPS}; do
  score_all_metrics "512_${step}" "${PROMPT_FILE}"
done
