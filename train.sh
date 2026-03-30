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
FLUX_FAKE_OUTDIR="${FLUX_FAKE_OUTDIR:-exps/fflux}"
FLUX_CONFIG="${FLUX_CONFIG:-training/ppo/cfgs/flux_dev.yaml}"
FLUX_EXPORT_RUN_NAME="${FLUX_EXPORT_RUN_NAME:-flux_dev}"
FLUX_EXPORT_CHECKPOINT="${FLUX_EXPORT_CHECKPOINT:-}"
FLUX_MODEL_REF="${FLUX_MODEL_PATH:-}"
if [[ -z "${FLUX_MODEL_REF}" ]]; then
  FLUX_MODEL_REF="$(resolve_local_flux_snapshot || true)"
fi
FLUX_MODEL_REF="${FLUX_MODEL_REF:-black-forest-labs/FLUX.1-dev}"
FLUX_BACKEND_OPTIONS="$(printf '{"model_name_or_path":"%s","torch_dtype":"bfloat16","enable_model_cpu_offload":false}' "${FLUX_MODEL_REF}")"

launch_with_snapshot() {
  local master_port="$1"
  local config_path="$2"
  local snapshot_path="$3"
  torchrun --master_port="${master_port}" --nproc_per_node=1 -m training.ppo.launch \
    --config "${config_path}" \
    --override "data.predictor_snapshot=${snapshot_path}"
}

python fake_train.py \
  --num-steps 11 \
  --num-points 2 \
  --outdir "${SD15_FAKE_OUTDIR}" \
  --r-base 0.5 \
  --r-epsilon 0.33 \
  --scale-dir 0.05 \
  --scale-time 0.05

launch_with_snapshot 12312 "${SD15_CONFIGS[0]}" "${SD15_FAKE_OUTDIR}/network-snapshot-000005.pkl"
launch_with_snapshot 23123 "${SD15_CONFIGS[1]}" "${SD15_FAKE_OUTDIR}/network-snapshot-000005.pkl"
launch_with_snapshot 31231 "${SD15_CONFIGS[2]}" "${SD15_FAKE_OUTDIR}/network-snapshot-000005.pkl"

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

launch_with_snapshot 33333 "${SD3_512_CONFIG}" "${SD3_512_FAKE_OUTDIR}/network-snapshot-000005.pkl"
launch_with_snapshot 44444 "${SD3_1024_CONFIG}" "${SD3_1024_FAKE_OUTDIR}/network-snapshot-000005.pkl"

python fake_train.py \
  --outdir "${FLUX_FAKE_OUTDIR}" \
  --num-steps 11 \
  --num-points 2 \
  --guidance-rate 3.5 \
  --schedule-type flowmatch \
  --backend flux \
  --resolution 1024 \
  --backend-options "${FLUX_BACKEND_OPTIONS}" \
  --r-base 0.5 \
  --r-epsilon 0.33

run_flux_runtime_preflight "${FLUX_MODEL_REF}"
launch_with_snapshot 55555 "${FLUX_CONFIG}" "${FLUX_FAKE_OUTDIR}/network-snapshot-000005.pkl"

SD3_512_EXPORT_RUN_DIR="${SD3_512_EXPORT_RUN_DIR:-$(latest_run_dir "${SD3_512_EXPORT_RUN_NAME}" || true)}"
if [[ -n "${SD3_512_EXPORT_RUN_DIR}" ]]; then
  run_export_predictor "${SD3_512_EXPORT_RUN_DIR}" "${SD3_512_EXPORT_CHECKPOINT}"
else
  echo "[train.sh] Skip SD3-512 export because no run dir matched '*-${SD3_512_EXPORT_RUN_NAME}'." >&2
fi

FLUX_EXPORT_RUN_DIR="${FLUX_EXPORT_RUN_DIR:-$(latest_run_dir "${FLUX_EXPORT_RUN_NAME}" || true)}"
if [[ -n "${FLUX_EXPORT_RUN_DIR}" ]]; then
  run_export_predictor "${FLUX_EXPORT_RUN_DIR}" "${FLUX_EXPORT_CHECKPOINT}"
else
  echo "[train.sh] Skip FLUX export because no run dir matched '*-${FLUX_EXPORT_RUN_NAME}'." >&2
fi
