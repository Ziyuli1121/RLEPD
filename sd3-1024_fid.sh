#!/usr/bin/env bash
set -euo pipefail

CONDA_BASE="${CONDA_EXE:-}"
CONDA_BASE="${CONDA_BASE%/bin/conda}"
if [[ -z "${CONDA_BASE}" || ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    CONDA_BASE="$(conda info --base)"
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda activate epd
cd /work/nvme/betk/zli42/RLEPD

CUDA_VISIBLE_DEVICES=1 python sample_sd3.py \
  --predictor exps/best_models/sd3-1024/sd3-1024-best.pkl \
  --prompt-file src/prompts/coco10k.txt \
  --seeds 0-9999 \
  --max-batch-size 2 \
  --outdir samples/sd3_1024_best_coco10k

CUDA_VISIBLE_DEVICES=0 python sample_sd3_baseline.py \
  --sampler sd3 \
  --resolution 1024 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file src/prompts/coco10k.txt \
  --seeds 0-9999 \
  --batch 4 \
  --num-steps 28 \
  --outdir samples/sd3_1024_euler_coco10k
