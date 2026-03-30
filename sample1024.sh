#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/pipeline_common.sh"

PROMPT_FILE="${PROMPT_FILE:-${RLEPD_PROMPTS_TXT_DEFAULT}}"
RUN_DIR="${RUN_DIR:-exps/20251206-131339-sd3_1024}"
EXPORT_STEP="${EXPORT_STEP:-7000}"

require_dir "${RUN_DIR}"
require_file "${PROMPT_FILE}"

run_export_predictor "${RUN_DIR}" "checkpoints/policy-step$(printf '%06d' "${EXPORT_STEP}").pt"
PREDICTOR="$(resolve_export_predictor "${RUN_DIR}" "${EXPORT_STEP}")"
require_file "${PREDICTOR}"

python sample_sd3.py \
  --predictor "${PREDICTOR}" \
  --prompt-file "${PROMPT_FILE}" \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir "samples/sd3_epd_1024_${EXPORT_STEP}"

python sample_sd3_baseline.py --sampler edm --resolution 1024 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file "${PROMPT_FILE}" \
  --seeds "0-999" --batch 8 \
  --num-steps 10 \
  --outdir ./samples/sd3_edm_20_1024

python sample_sd3_baseline.py --sampler dpm2 --resolution 1024 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file "${PROMPT_FILE}" \
  --seeds "0-999" --batch 8 \
  --num-steps 10 \
  --outdir ./samples/sd3_dpm2_20_1024

python sample_sd3_baseline.py --sampler ipndm --resolution 1024 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file "${PROMPT_FILE}" \
  --seeds "0-999" --batch 8 \
  --num-steps 20 --max-order 4 \
  --outdir ./samples/sd3_ipndm4_20_1024

python sample_sd3_baseline.py --sampler sd3 --resolution 1024 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file "${PROMPT_FILE}" \
  --seeds "0-999" --batch 8 \
  --num-steps 20 \
  --outdir ./samples/sd3_default_20_1024_nofinal

python sample_sd3_baseline.py --sampler dpm2 --resolution 1024 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file "${PROMPT_FILE}" \
  --seeds "0-999" --batch 8 \
  --num-steps 14 \
  --outdir ./samples/sd3_dpm2_28_1024

python sample_sd3_baseline.py --sampler ipndm --resolution 1024 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file "${PROMPT_FILE}" \
  --seeds "0-999" --batch 8 \
  --num-steps 28 --max-order 4 \
  --outdir ./samples/sd3_ipndm4_28_1024

score_all_metrics sd3_edm_20_1024 "${PROMPT_FILE}"
score_all_metrics sd3_dpm2_20_1024 "${PROMPT_FILE}"
score_all_metrics sd3_ipndm4_20_1024 "${PROMPT_FILE}"
score_all_metrics sd3_default_20_1024_nofinal "${PROMPT_FILE}"
score_all_metrics sd3_dpm2_28_1024 "${PROMPT_FILE}"
score_all_metrics sd3_ipndm4_28_1024 "${PROMPT_FILE}"
score_all_metrics "sd3_epd_1024_${EXPORT_STEP}" "${PROMPT_FILE}"
