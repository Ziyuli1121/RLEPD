#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/pipeline_common.sh"

SD15_FAKE_OUTDIR="${SD15_FAKE_OUTDIR:-exps/f15}"
SD15_CONFIGS=(
    "training/ppo/cfgs/sd15_k5.yaml"
    "training/ppo/cfgs/sd15_k20.yaml"
    "training/ppo/cfgs/sd15_k50.yaml"
)
SD15_EXPORT_RUN_NAME="${SD15_EXPORT_RUN_NAME:-sd15_k20}"
SD15_EXPORT_CHECKPOINT="${SD15_EXPORT_CHECKPOINT:-}"

SD3_512_FAKE_OUTDIR="${SD3_512_FAKE_OUTDIR:-exps/f512}"
SD3_1024_FAKE_OUTDIR="${SD3_1024_FAKE_OUTDIR:-exps/f1024}"
SD3_512_CONFIG="${SD3_512_CONFIG:-training/ppo/cfgs/sd3_512.yaml}"
SD3_1024_CONFIG="${SD3_1024_CONFIG:-training/ppo/cfgs/sd3_1024.yaml}"
SD3_512_EXPORT_RUN_NAME="${SD3_512_EXPORT_RUN_NAME:-sd3_512_new}"
SD3_512_EXPORT_CHECKPOINT="${SD3_512_EXPORT_CHECKPOINT:-}"

python fake_train.py \
  --num-steps 11 \
  --num-points 2 \
  --outdir "${SD15_FAKE_OUTDIR}" \
  --r-base 0.5 \
  --r-epsilon 0.33 \
  --scale-dir 0.05 \
  --scale-time 0.05

torchrun --master_port=12312 --nproc_per_node=1 -m training.ppo.launch --config "${SD15_CONFIGS[0]}"
torchrun --master_port=23123 --nproc_per_node=1 -m training.ppo.launch --config "${SD15_CONFIGS[1]}"
torchrun --master_port=31231 --nproc_per_node=1 -m training.ppo.launch --config "${SD15_CONFIGS[2]}"

SD15_EXPORT_RUN_DIR="${SD15_EXPORT_RUN_DIR:-$(latest_run_dir "${SD15_EXPORT_RUN_NAME}" || true)}"
if [[ -n "${SD15_EXPORT_RUN_DIR}" ]]; then
  run_export_predictor "${SD15_EXPORT_RUN_DIR}" "${SD15_EXPORT_CHECKPOINT}"
else
  echo "[train.sh] Skip SD1.5 export because no run dir matched '*-${SD15_EXPORT_RUN_NAME}'." >&2
fi

python fake_train.py \
  --outdir "${SD3_512_FAKE_OUTDIR}" \
  --num-steps 11 \
  --num-points 2 \
  --guidance-rate 4.5 \
  --schedule-type flowmatch \
  --backend sd3 \
  --resolution 512 \
  --backend-options '{"model_name_or_path":"stabilityai/stable-diffusion-3-medium-diffusers"}' \
  --r-base 0.5 \
  --r-epsilon 0.33

python fake_train.py \
  --outdir "${SD3_1024_FAKE_OUTDIR}" \
  --num-steps 11 \
  --num-points 2 \
  --guidance-rate 4.5 \
  --schedule-type flowmatch \
  --backend sd3 \
  --resolution 1024 \
  --backend-options '{"model_name_or_path":"stabilityai/stable-diffusion-3-medium-diffusers"}' \
  --r-base 0.5 \
  --r-epsilon 0.33

torchrun --master_port=33333 --nproc_per_node=1 -m training.ppo.launch --config "${SD3_512_CONFIG}"
torchrun --master_port=44444 --nproc_per_node=1 -m training.ppo.launch --config "${SD3_1024_CONFIG}"

SD3_512_EXPORT_RUN_DIR="${SD3_512_EXPORT_RUN_DIR:-$(latest_run_dir "${SD3_512_EXPORT_RUN_NAME}" || true)}"
if [[ -n "${SD3_512_EXPORT_RUN_DIR}" ]]; then
  run_export_predictor "${SD3_512_EXPORT_RUN_DIR}" "${SD3_512_EXPORT_CHECKPOINT}"
else
  echo "[train.sh] Skip SD3-512 export because no run dir matched '*-${SD3_512_EXPORT_RUN_NAME}'." >&2
fi
