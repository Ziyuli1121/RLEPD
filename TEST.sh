#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT}/scripts/pipeline_common.sh"
cd "${ROOT}"

log() {
    printf '\n[TEST] %s\n' "$*"
}

die() {
    printf '\n[TEST][ERROR] %s\n' "$*" >&2
    exit 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"
}

require_file() {
    [[ -f "$1" ]] || die "Missing file: $1"
}

require_dir() {
    [[ -d "$1" ]] || die "Missing directory: $1"
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

normalize_backbones() {
    local raw="${1,,}"
    raw="${raw// /,}"
    while [[ "${raw}" == *",,"* ]]; do
        raw="${raw//,,/,}"
    done
    raw="${raw#,}"
    raw="${raw%,}"
    printf '%s\n' "${raw:-all}"
}

backbone_enabled() {
    local needle="$1"
    if [[ "${BACKBONES}" == "all" ]]; then
        return 0
    fi
    [[ ",${BACKBONES}," == *",${needle},"* ]]
}

count_pngs() {
    find "$1" -type f -name '*.png' | wc -l | tr -d ' '
}

require_nonblack_png() {
    local image_path="$1"
    python - "${image_path}" <<'PY'
from PIL import Image
import numpy as np
import sys

path = sys.argv[1]
arr = np.array(Image.open(path).convert("RGB"))
if int(arr.max()) <= 0:
    raise SystemExit(f"Image is pure black: {path}")
PY
}

latest_run_dir() {
    local root="$1"
    local run_name="$2"
    find "${root}" -maxdepth 1 -mindepth 1 -type d -name "*-${run_name}" | sort | tail -n 1
}

latest_export_predictor() {
    local run_dir="$1"
    find "${run_dir}/export" -maxdepth 1 -type f -name 'network-snapshot-export-*.pkl' | sort | tail -n 1
}

require_cmd python
require_cmd torchrun

BACKBONES="$(normalize_backbones "${BACKBONES:-all}")"
log "Enabled backbones: ${BACKBONES}"

python - <<'PY'
import os
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for TEST.sh")

device_name = torch.cuda.get_device_name(0)
print(f"[TEST] python={os.sys.version.split()[0]} torch={torch.__version__} device={device_name}")
PY

export HF_HOME="${HF_HOME:-${ROOT}/weights/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

require_file "${ROOT}/weights/HPS_v2.1_compressed.pt"
require_file "${ROOT}/weights/ImageReward.pt"
require_file "${ROOT}/weights/sac+logos+ava1-l14-linearMSE.pth"
require_file "${ROOT}/weights/med_config.json"
require_dir "${ROOT}/weights/clip"
if backbone_enabled "sd15"; then
    require_file "${ROOT}/src/ms_coco/v1-5-pruned-emaonly.ckpt"
    require_file "${ROOT}/weights/MPS_overall_checkpoint.pth"
    require_dir "${ROOT}/MPS"
fi

resolve_cached_sd3_model() {
    python - <<'PY'
from pathlib import Path

repo_root = Path.cwd()
candidates = []

hf_home = Path.home() / ".cache" / "huggingface" / "hub"
candidates.extend(sorted(hf_home.glob("models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/*")))

local_hf = repo_root / "weights" / "hf_cache" / "hub"
candidates.extend(sorted(local_hf.glob("models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/*")))

for path in candidates:
    if (path / "model_index.json").is_file():
        print(path.resolve())
        raise SystemExit(0)
PY
}

resolve_cached_flux_model() {
    python - <<'PY'
from pathlib import Path

repo_root = Path.cwd()
candidates = []

hf_home = Path.home() / ".cache" / "huggingface" / "hub"
candidates.extend(sorted(hf_home.glob("models--black-forest-labs--FLUX.1-dev/snapshots/*")))

local_hf = repo_root / "weights" / "hf_cache" / "hub"
candidates.extend(sorted(local_hf.glob("models--black-forest-labs--FLUX.1-dev/snapshots/*")))

for path in candidates:
    if (path / "model_index.json").is_file():
        print(path.resolve())
        raise SystemExit(0)
PY
}

SD3_MODEL_REF=""
if backbone_enabled "sd3_512" || backbone_enabled "sd3_1024"; then
    if [[ -n "${SD3_MODEL_PATH:-}" ]]; then
        [[ -e "${SD3_MODEL_PATH}" ]] || die "SD3_MODEL_PATH is set but does not exist: ${SD3_MODEL_PATH}"
        SD3_MODEL_REF="${SD3_MODEL_PATH}"
    else
        SD3_MODEL_REF="$(resolve_cached_sd3_model || true)"
        if [[ -z "${SD3_MODEL_REF}" ]]; then
            if [[ "${SD3_ALLOW_REMOTE:-0}" == "1" ]]; then
                SD3_MODEL_REF="stabilityai/stable-diffusion-3-medium-diffusers"
            else
                die "SD3 model is unavailable locally. Set SD3_MODEL_PATH to a local Stable Diffusion 3 diffusers snapshot, or run with SD3_ALLOW_REMOTE=1 after authenticating to Hugging Face for the gated repo."
            fi
        fi
    fi
    log "Using SD3 model reference: ${SD3_MODEL_REF}"
fi

FLUX_MODEL_REF=""
if backbone_enabled "flux"; then
    if [[ -n "${FLUX_MODEL_PATH:-}" ]]; then
        [[ -e "${FLUX_MODEL_PATH}" ]] || die "FLUX_MODEL_PATH is set but does not exist: ${FLUX_MODEL_PATH}"
        FLUX_MODEL_REF="${FLUX_MODEL_PATH}"
    else
        FLUX_MODEL_REF="$(resolve_cached_flux_model || true)"
        if [[ -z "${FLUX_MODEL_REF}" ]]; then
            if [[ "${FLUX_ALLOW_REMOTE:-0}" == "1" ]]; then
                FLUX_MODEL_REF="black-forest-labs/FLUX.1-dev"
            else
                die "FLUX model is unavailable locally. Set FLUX_MODEL_PATH to a local FLUX.1-dev diffusers snapshot, or run with FLUX_ALLOW_REMOTE=1 after authenticating to Hugging Face for the gated repo."
            fi
        fi
    fi
    log "Using FLUX model reference: ${FLUX_MODEL_REF}"
fi

TEST_ID="${TEST_ID:-$(date +%Y%m%d-%H%M%S)}"
SMOKE_ROOT="${ROOT}/smoke_tests/${TEST_ID}"
PROMPTS_DIR="${SMOKE_ROOT}/prompts"
FAKE_ROOT="${SMOKE_ROOT}/fake"
RUNS_ROOT="${SMOKE_ROOT}/runs"
SAMPLES_ROOT="${SMOKE_ROOT}/samples"
RESULTS_ROOT="${SMOKE_ROOT}/results"
mkdir -p "${PROMPTS_DIR}" "${FAKE_ROOT}" "${RUNS_ROOT}" "${SAMPLES_ROOT}" "${RESULTS_ROOT}"

PROMPTS_5="${PROMPTS_DIR}/smoke_prompts_5.txt"
PROMPTS_2="${PROMPTS_DIR}/smoke_prompts_2.txt"
PROMPTS_1="${PROMPTS_DIR}/smoke_prompts_1.txt"

cat > "${PROMPTS_5}" <<'EOF'
a cinematic portrait of an astronaut reading a book in a greenhouse
a small red tram crossing a rainy bridge at dusk
a watercolor painting of a lighthouse on icy cliffs
a macro photo of a mechanical watch with glowing gears
a cozy living room with sunlight and lots of plants
EOF

cat > "${PROMPTS_2}" <<'EOF'
a cinematic portrait of an astronaut reading a book in a greenhouse
a small red tram crossing a rainy bridge at dusk
EOF

cat > "${PROMPTS_1}" <<'EOF'
a watercolor painting of a lighthouse on icy cliffs
EOF

SD15_NUM_STEPS="${SD15_NUM_STEPS:-4}"
SD3_512_NUM_STEPS="${SD3_512_NUM_STEPS:-4}"
SD3_1024_NUM_STEPS="${SD3_1024_NUM_STEPS:-3}"
FLUX_NUM_STEPS="${FLUX_NUM_STEPS:-3}"
NUM_POINTS="${NUM_POINTS:-2}"
TRAIN_STEPS="${TRAIN_STEPS:-10}"
TRAIN_STEP_TAG="$(printf '%06d' "${TRAIN_STEPS}")"

SD15_SEEDS="${SD15_SEEDS:-0-4}"
SD3_512_SEEDS="${SD3_512_SEEDS:-0-1}"
SD3_1024_SEEDS="${SD3_1024_SEEDS:-0}"
FLUX_SEEDS="${FLUX_SEEDS:-0}"
FLUX_SOLVER_SWEEP="${FLUX_SOLVER_SWEEP:-0}"
FLUX_BASELINE_SOLVERS="${FLUX_BASELINE_SOLVERS:-flux edm dpm dpm2 heun ipndm ddim}"

SMOKE_SD15_RUN_NAME="smoke_sd15"
SMOKE_SD3_512_RUN_NAME="smoke_sd3_512"
SMOKE_SD3_1024_RUN_NAME="smoke_sd3_1024"
SMOKE_FLUX_RUN_NAME="smoke_flux"

SD15_FAKE_DIR="${FAKE_ROOT}/f15"
SD3_512_FAKE_DIR="${FAKE_ROOT}/f512"
SD3_1024_FAKE_DIR="${FAKE_ROOT}/f1024"
FLUX_FAKE_DIR="${FAKE_ROOT}/fflux"

SD15_SAMPLE_DIR="${SAMPLES_ROOT}/sd15"
SD3_512_SAMPLE_DIR="${SAMPLES_ROOT}/sd3_512"
SD3_1024_SAMPLE_DIR="${SAMPLES_ROOT}/sd3_1024"
FLUX_SAMPLE_DIR="${SAMPLES_ROOT}/flux"
LAST_RUN_DIR=""
FLUX_PROMPTS_MATCHED="${PROMPTS_DIR}/smoke_prompts_flux_matched.txt"
FLUX_EXPECTED_COUNT="$(python - "${FLUX_SEEDS}" <<'PY'
from training.ppo.pipeline_utils import parse_seed_spec
import sys
print(len(parse_seed_spec(sys.argv[1])))
PY
)"

SD3_BACKEND_OPTIONS="$(SD3_MODEL_REF="${SD3_MODEL_REF}" python - <<'PY'
import json
import os

print(json.dumps({
    "model_name_or_path": os.environ["SD3_MODEL_REF"],
    "torch_dtype": "float16",
}))
PY
)"

FLUX_BACKEND_OPTIONS="$(FLUX_MODEL_REF="${FLUX_MODEL_REF}" python - <<'PY'
import json
import os

print(json.dumps({
    "model_name_or_path": os.environ["FLUX_MODEL_REF"],
    "torch_dtype": "bfloat16",
    "enable_model_cpu_offload": False,
}))
PY
)"

run_fake_train_sd15() {
    log "Stage 1: fake_train for sd1.5"
    python fake_train.py \
        --num-steps "${SD15_NUM_STEPS}" \
        --num-points "${NUM_POINTS}" \
        --outdir "${SD15_FAKE_DIR}" \
        --r-base 0.5 \
        --r-epsilon 0.25 \
        --scale-dir 0.05 \
        --scale-time 0.05 \
        --summary-json "${SD15_FAKE_DIR}/summary.json"

    require_file "${SD15_FAKE_DIR}/network-snapshot-000005.pkl"
    require_file "${SD15_FAKE_DIR}/training_options.json"
}

run_fake_train_sd3() {
    local outdir="$1"
    local resolution="$2"
    local num_steps="$3"

    log "Stage 1: fake_train for sd3-${resolution}"
    python fake_train.py \
        --outdir "${outdir}" \
        --num-steps "${num_steps}" \
        --num-points "${NUM_POINTS}" \
        --guidance-rate 4.5 \
        --schedule-type flowmatch \
        --backend sd3 \
        --resolution "${resolution}" \
        --backend-options "${SD3_BACKEND_OPTIONS}" \
        --r-base 0.5 \
        --r-epsilon 0.25 \
        --summary-json "${outdir}/summary.json"

    require_file "${outdir}/network-snapshot-000005.pkl"
    require_file "${outdir}/training_options.json"
}

run_fake_train_flux() {
    log "Stage 1: fake_train for flux"
    python fake_train.py \
        --outdir "${FLUX_FAKE_DIR}" \
        --num-steps "${FLUX_NUM_STEPS}" \
        --num-points "${NUM_POINTS}" \
        --guidance-rate 3.5 \
        --schedule-type flowmatch \
        --backend flux \
        --resolution 1024 \
        --backend-options "${FLUX_BACKEND_OPTIONS}" \
        --r-base 0.5 \
        --r-epsilon 0.25 \
        --summary-json "${FLUX_FAKE_DIR}/summary.json"

    require_file "${FLUX_FAKE_DIR}/network-snapshot-000005.pkl"
    require_file "${FLUX_FAKE_DIR}/training_options.json"
}

run_smoke_train() {
    local config="$1"
    local predictor_snapshot="$2"
    local run_name="$3"
    shift 3

    log "Stage 2: PPO smoke train for ${run_name}"
    local -a cmd=(
        torchrun --standalone --nproc_per_node=1 -m training.ppo.launch
        --config "${config}"
        --run-name "${run_name}"
        --max-steps "${TRAIN_STEPS}"
        --override "run.output_root=${RUNS_ROOT}"
        --override "data.predictor_snapshot=${predictor_snapshot}"
        --override "logging.save_interval=${TRAIN_STEPS}"
    )
    while [[ $# -gt 0 ]]; do
        cmd+=(--override "$1")
        shift
    done

    "${cmd[@]}"

    local run_dir
    run_dir="$(latest_run_dir "${RUNS_ROOT}" "${run_name}")"
    [[ -n "${run_dir}" ]] || die "Could not locate run dir for ${run_name}"
    require_dir "${run_dir}"
    require_file "${run_dir}/configs/resolved_config.yaml"
    require_file "${run_dir}/logs/metrics.jsonl"
    require_file "${run_dir}/checkpoints/policy-step${TRAIN_STEP_TAG}.pt"
    LAST_RUN_DIR="${run_dir}"
}

run_export() {
    local run_dir="$1"
    log "Stage 3: export predictor for ${run_dir}"
    python -m training.ppo.export_epd_predictor "${run_dir}"

    local predictor
    predictor="$(latest_export_predictor "${run_dir}")"
    [[ -n "${predictor}" ]] || die "Could not locate export predictor under ${run_dir}/export"
    require_file "${predictor}"
    require_file "${run_dir}/export/export-manifest-step${TRAIN_STEP_TAG}.json"
    require_file "${run_dir}/export/training_options-export-step${TRAIN_STEP_TAG}.json"
    printf '%s\n' "${predictor}"
}

run_sample_sd15() {
    local predictor_ref="$1"
    local outdir="$2"
    log "Stage 4: sample sd1.5"
    MASTER_ADDR=127.0.0.1 MASTER_PORT=29651 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
        python sample.py \
            --predictor_path "${predictor_ref}" \
            --prompt-file "${PROMPTS_5}" \
            --seeds "${SD15_SEEDS}" \
            --batch 1 \
            --outdir "${outdir}"

    local count
    count="$(count_pngs "${outdir}")"
    [[ "${count}" == "5" ]] || die "Expected 5 SD1.5 images, found ${count} in ${outdir}"
}

run_sample_sd3() {
    local predictor_ref="$1"
    local outdir="$2"
    local resolution="$3"
    local prompt_file="$4"
    local seeds="$5"
    local expected_count="$6"

    log "Stage 4: sample sd3-${resolution}"
    python sample_sd3.py \
        --predictor "${predictor_ref}" \
        --prompt-file "${prompt_file}" \
        --seeds "${seeds}" \
        --max-batch-size 1 \
        --resolution "${resolution}" \
        --outdir "${outdir}"

    local count
    count="$(count_pngs "${outdir}")"
    [[ "${count}" == "${expected_count}" ]] || die "Expected ${expected_count} SD3-${resolution} images, found ${count} in ${outdir}"
}

run_sample_flux() {
    local predictor_ref="$1"
    local outdir="$2"
    local prompt_file="$3"
    local seeds="$4"
    local expected_count="$5"

    log "Stage 4: sample flux"
    python sample_flux.py \
        --predictor "${predictor_ref}" \
        --prompt-file "${prompt_file}" \
        --seeds "${seeds}" \
        --max-batch-size 1 \
        --resolution 1024 \
        --outdir "${outdir}"

    local count
    count="$(count_pngs "${outdir}")"
    [[ "${count}" == "${expected_count}" ]] || die "Expected ${expected_count} FLUX images, found ${count} in ${outdir}"
}

run_flux_baseline_sweep() {
    log "Stage 4b: FLUX baseline solver sweep"
    local solver
    for solver in ${FLUX_BASELINE_SOLVERS}; do
        local solver_outdir="${SAMPLES_ROOT}/flux_baseline_${solver}"
        python sample_flux_baseline.py \
            --sampler "${solver}" \
            --model-id "${FLUX_MODEL_REF}" \
            --prompt-file "${FLUX_PROMPTS_MATCHED}" \
            --seeds "${FLUX_SEEDS}" \
            --batch 1 \
            --outdir "${solver_outdir}"

        local count
        count="$(count_pngs "${solver_outdir}")"
        [[ "${count}" == "${FLUX_EXPECTED_COUNT}" ]] || die "Expected ${FLUX_EXPECTED_COUNT} FLUX baseline images for ${solver}, found ${count} in ${solver_outdir}"
        local first_png
        first_png="$(find "${solver_outdir}" -type f -name '*.png' | sort | head -n 1)"
        [[ -n "${first_png}" ]] || die "Could not resolve baseline PNG for ${solver} under ${solver_outdir}"
        require_nonblack_png "${first_png}"
    done
}

run_metric() {
    local module="$1"
    local images_dir="$2"
    local prompt_file="$3"
    local weights_path="$4"
    local output_json="$5"

    python -m "${module}" \
        --images "${images_dir}" \
        --pattern "**/*.png" \
        --prompts "${prompt_file}" \
        --weights "${weights_path}" \
        --batch-size 1 \
        --output-json "${output_json}"
}

run_full_eval_sd15() {
    log "Stage 5: full eval suite on sd1.5 samples"
    run_metric training.ppo.scripts.score_clip \
        "${SD15_SAMPLE_DIR}" "${PROMPTS_5}" "${ROOT}/weights/clip" "${RESULTS_ROOT}/sd15_clip.json"
    run_metric training.ppo.scripts.score_hps \
        "${SD15_SAMPLE_DIR}" "${PROMPTS_5}" "${ROOT}/weights/HPS_v2.1_compressed.pt" "${RESULTS_ROOT}/sd15_hps.json"
    run_metric training.ppo.scripts.score_aesthetic \
        "${SD15_SAMPLE_DIR}" "${PROMPTS_5}" "${ROOT}/weights/sac+logos+ava1-l14-linearMSE.pth" "${RESULTS_ROOT}/sd15_aesthetic.json"
    run_metric training.ppo.scripts.score_imagereward \
        "${SD15_SAMPLE_DIR}" "${PROMPTS_5}" "${ROOT}/weights/ImageReward.pt" "${RESULTS_ROOT}/sd15_imagereward.json"
    run_metric training.ppo.scripts.score_mps \
        "${SD15_SAMPLE_DIR}" "${PROMPTS_5}" "${ROOT}/weights/MPS_overall_checkpoint.pth" "${RESULTS_ROOT}/sd15_mps.json"

    if [[ -d "${ROOT}/weights/PickScore_v1" || "${ENABLE_PICKSCORE_DOWNLOAD:-0}" == "1" ]]; then
        run_metric training.ppo.scripts.score_pick \
            "${SD15_SAMPLE_DIR}" "${PROMPTS_5}" "${ROOT}/weights/PickScore_v1" "${RESULTS_ROOT}/sd15_pick.json"
    else
        log "Skipping PickScore: local weights/PickScore_v1 is missing. Set ENABLE_PICKSCORE_DOWNLOAD=1 to force HF download."
    fi

    require_file "${RESULTS_ROOT}/sd15_clip.json"
    require_file "${RESULTS_ROOT}/sd15_hps.json"
    require_file "${RESULTS_ROOT}/sd15_aesthetic.json"
    require_file "${RESULTS_ROOT}/sd15_imagereward.json"
    require_file "${RESULTS_ROOT}/sd15_mps.json"
}

run_light_eval_sd3() {
    local prefix="$1"
    local images_dir="$2"
    local prompt_file="$3"
    log "Stage 5: lightweight eval on ${prefix}"
    run_metric training.ppo.scripts.score_hps \
        "${images_dir}" "${prompt_file}" "${ROOT}/weights/HPS_v2.1_compressed.pt" "${RESULTS_ROOT}/${prefix}_hps.json"
    require_file "${RESULTS_ROOT}/${prefix}_hps.json"
}

run_light_eval_flux() {
    local prompt_file="$1"
    log "Stage 5: lightweight eval on flux"
    run_metric training.ppo.scripts.score_hps \
        "${FLUX_SAMPLE_DIR}" "${prompt_file}" "${ROOT}/weights/HPS_v2.1_compressed.pt" "${RESULTS_ROOT}/flux_hps.json"
    require_file "${RESULTS_ROOT}/flux_hps.json"
}

if backbone_enabled "sd15"; then
    run_fake_train_sd15
    run_smoke_train \
        "training/ppo/cfgs/sd15_base.yaml" \
        "${SD15_FAKE_DIR}/network-snapshot-000005.pkl" \
        "${SMOKE_SD15_RUN_NAME}"
    SD15_RUN_DIR="${LAST_RUN_DIR}"
    run_export "${SD15_RUN_DIR}" >/dev/null
    run_sample_sd15 "${SD15_RUN_DIR}" "${SD15_SAMPLE_DIR}"
    run_full_eval_sd15
fi

if backbone_enabled "sd3_512"; then
    run_fake_train_sd3 "${SD3_512_FAKE_DIR}" 512 "${SD3_512_NUM_STEPS}"
    run_smoke_train \
        "training/ppo/cfgs/sd3_512.yaml" \
        "${SD3_512_FAKE_DIR}/network-snapshot-000005.pkl" \
        "${SMOKE_SD3_512_RUN_NAME}"
    SD3_512_RUN_DIR="${LAST_RUN_DIR}"
    run_export "${SD3_512_RUN_DIR}" >/dev/null
    run_sample_sd3 "${SD3_512_RUN_DIR}" "${SD3_512_SAMPLE_DIR}" 512 "${PROMPTS_2}" "${SD3_512_SEEDS}" 2
    run_light_eval_sd3 "sd3_512" "${SD3_512_SAMPLE_DIR}" "${PROMPTS_2}"
fi

if backbone_enabled "sd3_1024"; then
    run_fake_train_sd3 "${SD3_1024_FAKE_DIR}" 1024 "${SD3_1024_NUM_STEPS}"
    run_smoke_train \
        "training/ppo/cfgs/sd3_1024.yaml" \
        "${SD3_1024_FAKE_DIR}/network-snapshot-000005.pkl" \
        "${SMOKE_SD3_1024_RUN_NAME}"
    SD3_1024_RUN_DIR="${LAST_RUN_DIR}"
    run_export "${SD3_1024_RUN_DIR}" >/dev/null
    run_sample_sd3 "${SD3_1024_RUN_DIR}" "${SD3_1024_SAMPLE_DIR}" 1024 "${PROMPTS_1}" "${SD3_1024_SEEDS}" 1
    run_light_eval_sd3 "sd3_1024" "${SD3_1024_SAMPLE_DIR}" "${PROMPTS_1}"
fi

if backbone_enabled "flux"; then
    run_flux_runtime_preflight "${FLUX_MODEL_REF}"
    prepare_prompt_subset_file "${PROMPTS_1}" "${FLUX_PROMPTS_MATCHED}" "${FLUX_SEEDS}"
    run_fake_train_flux
    run_smoke_train \
        "training/ppo/cfgs/flux_dev.yaml" \
        "${FLUX_FAKE_DIR}/network-snapshot-000005.pkl" \
        "${SMOKE_FLUX_RUN_NAME}"
    FLUX_RUN_DIR="${LAST_RUN_DIR}"
    run_export "${FLUX_RUN_DIR}" >/dev/null
    run_sample_flux "${FLUX_RUN_DIR}" "${FLUX_SAMPLE_DIR}" "${FLUX_PROMPTS_MATCHED}" "${FLUX_SEEDS}" "${FLUX_EXPECTED_COUNT}"
    run_light_eval_flux "${FLUX_PROMPTS_MATCHED}"
    if [[ "${FLUX_SOLVER_SWEEP}" == "1" ]]; then
        run_flux_baseline_sweep
    fi
fi

log "Smoke test finished successfully."
log "Outputs:"
log "  smoke root: ${SMOKE_ROOT}"
log "  runs:       ${RUNS_ROOT}"
log "  samples:    ${SAMPLES_ROOT}"
log "  results:    ${RESULTS_ROOT}"
