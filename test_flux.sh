#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/pipeline_common.sh"

RUN_NAME="${RUN_NAME:-flux_dev}"
RUN_DIR="${RUN_DIR:-$(latest_run_dir "${RUN_NAME}" || true)}"
PROMPT_FILE="${PROMPT_FILE:-${RLEPD_PROMPTS_TXT_DEFAULT}}"
OUTDIR_NAME="${OUTDIR_NAME:-test_flux}"
BASELINE_OUTDIR_NAME="${BASELINE_OUTDIR_NAME:-test_flux_baseline}"
BASELINE_SOLVERS="${BASELINE_SOLVERS:-flux edm dpm dpm2 heun ipndm ddim}"
MODEL_ID="${MODEL_ID:-${FLUX_MODEL_PATH:-}}"
SEEDS="${SEEDS:-0}"

count_pngs() {
  find "$1" -type f -name '*.png' | wc -l | tr -d ' '
}

require_nonblack_png() {
  python - "$1" <<'PY'
from PIL import Image
import numpy as np
import sys

arr = np.array(Image.open(sys.argv[1]).convert("RGB"))
if int(arr.max()) <= 0:
    raise SystemExit(f"Image is pure black: {sys.argv[1]}")
PY
}

if [[ -z "${RUN_DIR}" ]]; then
  echo "[test_flux.sh] Could not resolve run dir for ${RUN_NAME}" >&2
  exit 1
fi

run_export_predictor "${RUN_DIR}"
PREDICTOR="$(resolve_export_predictor "${RUN_DIR}" || true)"
if [[ -z "${PREDICTOR}" ]]; then
  echo "[test_flux.sh] Could not resolve exported predictor." >&2
  exit 1
fi

if [[ -z "${MODEL_ID}" ]]; then
  MODEL_ID="$(resolve_local_flux_snapshot || true)"
fi
MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-dev}"
run_flux_runtime_preflight "${MODEL_ID}" "${PREDICTOR}"

mkdir -p "${RLEPD_RESULTS_DIR_DEFAULT}"
MATCHED_PROMPT_FILE="${RLEPD_RESULTS_DIR_DEFAULT}/${OUTDIR_NAME}_prompts.txt"
prepare_prompt_subset_file "${PROMPT_FILE}" "${MATCHED_PROMPT_FILE}" "${SEEDS}"
EXPECTED_COUNT="$(python - "${SEEDS}" <<'PY'
from training.ppo.pipeline_utils import parse_seed_spec
import sys
print(len(parse_seed_spec(sys.argv[1])))
PY
)"

python sample_flux.py \
  --predictor "${PREDICTOR}" \
  --prompt-file "${MATCHED_PROMPT_FILE}" \
  --seeds "${SEEDS}" \
  --max-batch-size 1 \
  --outdir "samples/${OUTDIR_NAME}"

score_all_metrics_dir "samples/${OUTDIR_NAME}" "${MATCHED_PROMPT_FILE}" "${OUTDIR_NAME}" "${RLEPD_RESULTS_DIR_DEFAULT}"
require_file "${RLEPD_RESULTS_DIR_DEFAULT}/${OUTDIR_NAME}_clip.json"
require_file "${RLEPD_RESULTS_DIR_DEFAULT}/${OUTDIR_NAME}_hps.json"
require_file "${RLEPD_RESULTS_DIR_DEFAULT}/${OUTDIR_NAME}_aesthetic.json"
require_file "${RLEPD_RESULTS_DIR_DEFAULT}/${OUTDIR_NAME}_imagereward.json"
require_file "${RLEPD_RESULTS_DIR_DEFAULT}/${OUTDIR_NAME}_mps.json"

for solver in ${BASELINE_SOLVERS}; do
  solver_outdir="samples/${BASELINE_OUTDIR_NAME}_${solver}"
  python sample_flux_baseline.py \
    --sampler "${solver}" \
    --model-id "${MODEL_ID}" \
    --prompt-file "${MATCHED_PROMPT_FILE}" \
    --seeds "${SEEDS}" \
    --batch 1 \
    --outdir "${solver_outdir}"
  if [[ "$(count_pngs "${solver_outdir}")" != "${EXPECTED_COUNT}" ]]; then
    echo "[test_flux.sh] Expected ${EXPECTED_COUNT} PNGs for solver ${solver} under ${solver_outdir}" >&2
    exit 1
  fi
  first_png="$(find "${solver_outdir}" -type f -name '*.png' | sort | head -n 1)"
  if [[ -z "${first_png}" ]]; then
    echo "[test_flux.sh] Could not resolve PNG for solver ${solver} under ${solver_outdir}" >&2
    exit 1
  fi
  require_nonblack_png "${first_png}"
done
