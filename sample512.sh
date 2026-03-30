#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/pipeline_common.sh"

PROMPT_FILE="${PROMPT_FILE:-${RLEPD_PROMPTS_TXT_DEFAULT}}"
RUN_DIR="${RUN_DIR:-exps/20251201-201759-sd3_512}"
STEPS="${STEPS:-500 1000 1500 2450}"

require_dir "${RUN_DIR}"
require_file "${PROMPT_FILE}"

for step in ${STEPS}; do
  predictor="${RUN_DIR}/export/network-snapshot-export-step$(printf '%06d' "${step}").pkl"
  require_file "${predictor}"
  python sample_sd3.py \
    --predictor "${predictor}" \
    --prompt-file "${PROMPT_FILE}" \
    --seeds "0-999" \
    --max-batch-size 4 \
    --outdir "samples/sd3_epd_9_512_${step}"
done

# Baseline examples can be enabled by uncommenting the commands below.
# python sample_sd3_baseline.py --sampler edm --resolution 512 --model-id "stabilityai/stable-diffusion-3-medium-diffusers" --prompt-file "${PROMPT_FILE}" --seeds "0-999" --batch 8 --num-steps 14 --outdir ./samples/sd3_edm_flowmatch_28_512
# python sample_sd3_baseline.py --sampler dpm2 --resolution 512 --model-id "stabilityai/stable-diffusion-3-medium-diffusers" --prompt-file "${PROMPT_FILE}" --seeds "0-999" --batch 8 --num-steps 15 --outdir ./samples/sd3_dpm2_flowmatch_28_512
# python sample_sd3_baseline.py --sampler ipndm --resolution 512 --model-id "stabilityai/stable-diffusion-3-medium-diffusers" --prompt-file "${PROMPT_FILE}" --seeds "0-999" --batch 8 --num-steps 29 --max-order 3 --outdir ./samples/sd3_ipndm_flowmatch_28_512
# python sample_sd3_baseline.py --sampler sd3 --resolution 512 --model-id "stabilityai/stable-diffusion-3-medium-diffusers" --prompt-file "${PROMPT_FILE}" --seeds "0-999" --batch 8 --num-steps 28 --outdir ./samples/sd3_default_28_512

score_all_metrics sd3_epd_9_512_1000 "${PROMPT_FILE}"
score_all_metrics sd3_epd_9_512_1500 "${PROMPT_FILE}"
score_all_metrics sd3_epd_9_512_2450 "${PROMPT_FILE}"
