#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/pipeline_common.sh"

PROMPT_FILE="${PROMPT_FILE:-${RLEPD_PROMPTS_TXT_DEFAULT}}"

# SD3 baseline examples
python sample_sd3_baseline.py --sampler edm --resolution 512 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file "${PROMPT_FILE}" \
  --seeds "0-99" --batch 8 \
  --num-steps 2 \
  --outdir ./samples/sd3_edm_28_512

python sample_sd3_baseline.py --sampler dpm2 --resolution 512 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file "${PROMPT_FILE}" \
  --seeds "0-999" --batch 8 \
  --num-steps 14 \
  --outdir ./samples/sd3_dpm2_28_512

python sample_sd3_baseline.py --sampler ipndm --resolution 512 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file "${PROMPT_FILE}" \
  --seeds "0-999" --batch 8 \
  --num-steps 28 --max-order 4 \
  --outdir ./samples/sd3_ipndm4_28_512

python sample_sd3_baseline.py --sampler sd3 --resolution 512 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file "${PROMPT_FILE}" \
  --seeds "0-999" --batch 8 \
  --num-steps 28 \
  --outdir ./samples/sd3_default_28_512

score_all_metrics sd3_ipndm4_28_512 "${PROMPT_FILE}"
score_all_metrics sd3_default_28_512 "${PROMPT_FILE}"
score_all_metrics sd3_dpm2_28_512 "${PROMPT_FILE}"
score_all_metrics sd3_edm_28_512 "${PROMPT_FILE}"
