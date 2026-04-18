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

CUDA_VISIBLE_DEVICES=1 python sample.py \
  --predictor_path exps/best_models/sd15/sd15-best.pkl \
  --prompt-file src/prompts/coco10k.txt \
  --seeds 0-9999 \
  --batch 8 \
  --outdir samples/sd15_best_coco10k

CUDA_VISIBLE_DEVICES=0 python sample_baseline.py \
  --sampler ddim \
  --prompt-file src/prompts/coco10k.txt \
  --seeds 0-9999 \
  --batch 8 \
  --num-steps 50 \
  --outdir samples/sd15_ddim_coco10k







CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29501 python sample_baseline.py \
  --sampler ddim \
  --prompt-file src/prompts/coco10k.txt \
  --seeds 0-4999 \
  --batch 8 \
  --num-steps 50 \
  --outdir samples/sd15_ddim_coco10k


CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29502 python sample_baseline.py \
  --sampler ddim \
  --prompt-file src/prompts/coco10k.txt \
  --seeds 5000-9999 \
  --batch 8 \
  --num-steps 50 \
  --outdir samples/sd15_ddim_coco10k

CUDA_VISIBLE_DEVICES=0 python sample_baseline.py \
  --sampler edm \
  --prompt-file src/prompts/coco10k.txt \
  --seeds 0-9999 \
  --batch 8 \
  --num-steps 25 \
  --outdir samples/sd15_edm_coco10k

CUDA_VISIBLE_DEVICES=0 python sample_baseline.py \
  --sampler dpm2 \
  --prompt-file src/prompts/coco10k.txt \
  --seeds 0-9999 \
  --batch 8 \
  --num-steps 25 \
  --outdir samples/sd15_dpm2_coco10k

CUDA_VISIBLE_DEVICES=0 python sample_baseline.py \
  --sampler ipndm \
  --prompt-file src/prompts/coco10k.txt \
  --seeds 0-9999 \
  --batch 8 \
  --num-steps 50 \
  --max-order 4 \
  --outdir samples/sd15_ipndm_coco10k


