#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/pipeline_common.sh"

RUN_NAME="${RUN_NAME:-flux_dev}"
RUN_DIR="${RUN_DIR:-$(latest_run_dir "${RUN_NAME}" || true)}"
PREDICTOR="${PREDICTOR:-}"
PROMPT_FILE="${PROMPT_FILE:-${RLEPD_PROMPTS_TXT_DEFAULT}}"
SEEDS="${SEEDS:-0-9}"
OUTDIR="${OUTDIR:-samples/flux}"

if [[ -z "${PREDICTOR}" && -n "${RUN_DIR}" ]]; then
  PREDICTOR="$(resolve_export_predictor "${RUN_DIR}" || true)"
fi

if [[ -z "${PREDICTOR}" ]]; then
  echo "[sample_flux.sh] No predictor resolved. Set RUN_DIR or PREDICTOR." >&2
  exit 1
fi

run_flux_runtime_preflight "" "${PREDICTOR}"

python sample_flux.py \
  --predictor "${PREDICTOR}" \
  --prompt-file "${PROMPT_FILE}" \
  --seeds "${SEEDS}" \
  --max-batch-size 1 \
  --outdir "${OUTDIR}"
