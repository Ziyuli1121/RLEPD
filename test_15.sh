#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/pipeline_common.sh"

RUN_DIR="${RUN_DIR:-exps/20251222-171726-sd15_k20}"
PROMPT_FILE="${PROMPT_FILE:-${RLEPD_PROMPTS_TXT_DEFAULT}}"
SEEDS="${SEEDS:-0-999}"
BATCH_SIZE="${BATCH_SIZE:-16}"
STEPS="${STEPS:-1000 2000 3000 4000 5000 6000 7000 8000 9000}"

require_dir "${RUN_DIR}"
require_file "${PROMPT_FILE}"

for step in ${STEPS}; do
  run_export_predictor "${RUN_DIR}" "checkpoints/policy-step$(printf '%06d' "${step}").pt"
  predictor="$(resolve_export_predictor "${RUN_DIR}" "${step}")"
  require_file "${predictor}"
  MASTER_PORT="$((20000 + step))" python sample.py \
    --predictor_path "${predictor}" \
    --prompt-file "${PROMPT_FILE}" \
    --seeds "${SEEDS}" \
    --batch "${BATCH_SIZE}" \
    --outdir "./samples/15_${step}"
done

for step in ${STEPS}; do
  score_all_metrics "15_${step}" "${PROMPT_FILE}"
done
