#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/pipeline_common.sh"

log() {
  printf '\n[train_flux.sh] %s\n' "$*"
}

FLUX_FAKE_OUTDIR="${FLUX_FAKE_OUTDIR:-exps/fflux}"
FLUX_CONFIG="${FLUX_CONFIG:-training/ppo/cfgs/flux_dev.yaml}"
FLUX_RUN_NAME="${FLUX_RUN_NAME:-flux_dev}"
FLUX_EXPORT_AFTER="${FLUX_EXPORT_AFTER:-1}"
FLUX_EXPORT_CHECKPOINT="${FLUX_EXPORT_CHECKPOINT:-}"

FLUX_MODEL_REF="${FLUX_MODEL_PATH:-}"
if [[ -z "${FLUX_MODEL_REF}" ]]; then
  FLUX_MODEL_REF="$(resolve_local_flux_snapshot || true)"
fi
FLUX_MODEL_REF="${FLUX_MODEL_REF:-black-forest-labs/FLUX.1-dev}"

FLUX_MASTER_PORT="${FLUX_MASTER_PORT:-11111}"
FLUX_NPROC_PER_NODE="${FLUX_NPROC_PER_NODE:-1}"
FLUX_CUDA_LAUNCH_BLOCKING="${FLUX_CUDA_LAUNCH_BLOCKING:-1}"
FLUX_TORCH_SHOW_CPP_STACKTRACES="${FLUX_TORCH_SHOW_CPP_STACKTRACES:-1}"
FLUX_NUM_STEPS="${FLUX_NUM_STEPS:-9}"
FLUX_NUM_POINTS="${FLUX_NUM_POINTS:-2}"
FLUX_GUIDANCE_RATE="${FLUX_GUIDANCE_RATE:-3.5}"
FLUX_R_BASE="${FLUX_R_BASE:-0.5}"
FLUX_R_EPSILON="${FLUX_R_EPSILON:-0.33}"
FLUX_PROMPT_CSV="${FLUX_PROMPT_CSV:-}"
FLUX_MAX_STEPS="${FLUX_MAX_STEPS:-}"

FLUX_BACKEND_OPTIONS="$(printf '{"model_name_or_path":"%s","torch_dtype":"bfloat16","enable_model_cpu_offload":false}' "${FLUX_MODEL_REF}")"
FLUX_SNAPSHOT_PATH="${FLUX_FAKE_OUTDIR}/network-snapshot-000005.pkl"

log "Using FLUX model reference: ${FLUX_MODEL_REF}"
log "Stage 1: fake_train -> ${FLUX_FAKE_OUTDIR}"
python fake_train.py \
  --outdir "${FLUX_FAKE_OUTDIR}" \
  --num-steps "${FLUX_NUM_STEPS}" \
  --num-points "${FLUX_NUM_POINTS}" \
  --guidance-rate "${FLUX_GUIDANCE_RATE}" \
  --schedule-type flowmatch \
  --backend flux \
  --resolution 1024 \
  --backend-options "${FLUX_BACKEND_OPTIONS}" \
  --r-base "${FLUX_R_BASE}" \
  --r-epsilon "${FLUX_R_EPSILON}"

require_file "${FLUX_SNAPSHOT_PATH}"

log "Stage 2: runtime preflight"
run_flux_runtime_preflight "${FLUX_MODEL_REF}" "${FLUX_SNAPSHOT_PATH}"

log "Stage 3: PPO launch"
if [[ "${FLUX_NPROC_PER_NODE}" == "1" ]]; then
  log "Single-GPU launch: python -m training.ppo.launch"
  launch_cmd=(
    python
    -m training.ppo.launch
    --config "${FLUX_CONFIG}"
    --override "run.run_name=${FLUX_RUN_NAME}"
    --override "data.predictor_snapshot=${FLUX_SNAPSHOT_PATH}"
  )
else
  log "Multi-GPU launch: torchrun --nproc_per_node=${FLUX_NPROC_PER_NODE}"
  launch_cmd=(
    torchrun
    --master_port="${FLUX_MASTER_PORT}"
    --nproc_per_node="${FLUX_NPROC_PER_NODE}"
    -m training.ppo.launch
    --config "${FLUX_CONFIG}"
    --override "run.run_name=${FLUX_RUN_NAME}"
    --override "data.predictor_snapshot=${FLUX_SNAPSHOT_PATH}"
  )
fi
if [[ -n "${FLUX_PROMPT_CSV}" ]]; then
  launch_cmd+=(--override "data.prompt_csv=${FLUX_PROMPT_CSV}")
fi
if [[ -n "${FLUX_MAX_STEPS}" ]]; then
  launch_cmd+=(--max-steps "${FLUX_MAX_STEPS}")
fi
CUDA_LAUNCH_BLOCKING="${FLUX_CUDA_LAUNCH_BLOCKING}" \
TORCH_SHOW_CPP_STACKTRACES="${FLUX_TORCH_SHOW_CPP_STACKTRACES}" \
"${launch_cmd[@]}"

if [[ "${FLUX_EXPORT_AFTER}" == "1" ]]; then
  FLUX_RUN_DIR="$(latest_run_dir "${FLUX_RUN_NAME}" || true)"
  if [[ -n "${FLUX_RUN_DIR}" ]]; then
    log "Stage 4: export predictor -> ${FLUX_RUN_DIR}"
    run_export_predictor "${FLUX_RUN_DIR}" "${FLUX_EXPORT_CHECKPOINT}"
    log "Latest run: ${FLUX_RUN_DIR}"
    log "Latest export: $(resolve_export_predictor "${FLUX_RUN_DIR}" || true)"
  else
    echo "[train_flux.sh] Skip export because no run dir matched '*-${FLUX_RUN_NAME}'." >&2
  fi
fi
