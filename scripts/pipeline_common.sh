#!/usr/bin/env bash

if [[ -n "${RLEPD_PIPELINE_COMMON_SH:-}" ]]; then
    return 0 2>/dev/null || exit 0
fi
RLEPD_PIPELINE_COMMON_SH=1

RLEPD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RLEPD_PROMPTS_TXT_DEFAULT="${RLEPD_ROOT}/src/prompts/test.txt"
RLEPD_PROMPTS_CSV_DEFAULT="${RLEPD_ROOT}/src/prompts/MS-COCO_val2014_30k_captions.csv"
RLEPD_RESULTS_DIR_DEFAULT="${RLEPD_ROOT}/results"

cd "${RLEPD_ROOT}" || exit 1

resolve_existing_path() {
    local candidate
    for candidate in "$@"; do
        if [[ -n "${candidate:-}" && -e "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

require_file() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "[RLEPD] Missing file: ${path}" >&2
        return 1
    fi
}

require_dir() {
    local path="$1"
    if [[ ! -d "${path}" ]]; then
        echo "[RLEPD] Missing directory: ${path}" >&2
        return 1
    fi
}

latest_run_dir() {
    local suffix="$1"
    find "${RLEPD_ROOT}/exps" -maxdepth 1 -mindepth 1 -type d -name "*-${suffix}" | sort | tail -n 1
}

latest_policy_checkpoint() {
    local run_dir="$1"
    find "${run_dir}/checkpoints" -maxdepth 1 -type f -name 'policy-step*.pt' | sort | tail -n 1
}

resolve_policy_checkpoint() {
    local run_dir="$1"
    local checkpoint="${2:-}"
    if [[ -n "${checkpoint}" ]]; then
        if [[ "${checkpoint}" = /* ]]; then
            printf '%s\n' "${checkpoint}"
        else
            printf '%s/%s\n' "${run_dir}" "${checkpoint}"
        fi
        return 0
    fi
    latest_policy_checkpoint "${run_dir}"
}

resolve_export_predictor() {
    local run_dir="$1"
    local step="${2:-}"
    local digits
    local candidate
    if [[ -n "${step}" ]]; then
        digits="$(printf '%06d' "${step}")"
        candidate="${run_dir}/export/network-snapshot-export-step${digits}.pkl"
        if [[ -f "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    fi
    find "${run_dir}/export" -maxdepth 1 -type f -name 'network-snapshot-export-*.pkl' | sort | tail -n 1
}

run_export_predictor() {
    local run_dir="$1"
    local checkpoint="${2:-}"
    require_dir "${run_dir}" || return 1
    if [[ -n "${checkpoint}" ]]; then
        checkpoint="$(resolve_policy_checkpoint "${run_dir}" "${checkpoint}")"
        require_file "${checkpoint}" || return 1
        python -m training.ppo.export_epd_predictor "${run_dir}" --checkpoint "${checkpoint#${run_dir}/}"
    else
        python -m training.ppo.export_epd_predictor "${run_dir}"
    fi
}

score_all_metrics() {
    local name="$1"
    local prompt_file="${2:-${RLEPD_PROMPTS_TXT_DEFAULT}}"
    if [[ -z "${name}" ]]; then
        echo "Usage: score_all_metrics <images_subdir_under_samples> [prompt_file]" >&2
        return 1
    fi

    local image_dir="${RLEPD_ROOT}/samples/${name}"
    local prefix="${name}"
    local clip_weights="${RLEPD_ROOT}/weights/clip"
    local hps_weights
    local aesthetic_weights
    local pickscore_weights
    local imagereward_weights
    local mps_weights

    require_dir "${image_dir}" || return 1
    require_file "${prompt_file}" || return 1

    hps_weights="$(resolve_existing_path "${RLEPD_ROOT}/weights/HPS_v2.1_compressed.pt")"
    aesthetic_weights="$(resolve_existing_path "${RLEPD_ROOT}/weights/sac+logos+ava1-l14-linearMSE.pth")"
    mps_weights="$(resolve_existing_path "${RLEPD_ROOT}/weights/MPS_overall_checkpoint.pth")"
    pickscore_weights="$(resolve_existing_path "${RLEPD_ROOT}/weights/PickScore_v1" || true)"
    imagereward_weights="$(resolve_existing_path "${RLEPD_ROOT}/weights/ImageReward.pt" "${RLEPD_ROOT}/weights/ImageReward-v1.0.pt" || true)"

    require_file "${hps_weights}" || return 1
    require_file "${aesthetic_weights}" || return 1
    require_file "${mps_weights}" || return 1
    mkdir -p "${RLEPD_RESULTS_DIR_DEFAULT}"

    if [[ -z "${pickscore_weights}" ]]; then
        pickscore_weights="${RLEPD_ROOT}/weights/PickScore_v1"
    fi
    if [[ -z "${imagereward_weights}" ]]; then
        imagereward_weights="${RLEPD_ROOT}/weights/ImageReward.pt"
    fi

    python -m training.ppo.scripts.score_clip \
        --images "${image_dir}" \
        --pattern "**/*.png" \
        --prompts "${prompt_file}" \
        --weights "${clip_weights}" \
        --output-json "${RLEPD_RESULTS_DIR_DEFAULT}/${prefix}_clip.json"

    python -m training.ppo.scripts.score_hps \
        --images "${image_dir}" \
        --pattern "**/*.png" \
        --prompts "${prompt_file}" \
        --weights "${hps_weights}" \
        --output-json "${RLEPD_RESULTS_DIR_DEFAULT}/${prefix}_hps.json"

    python -m training.ppo.scripts.score_aesthetic \
        --images "${image_dir}" \
        --pattern "**/*.png" \
        --prompts "${prompt_file}" \
        --weights "${aesthetic_weights}" \
        --output-json "${RLEPD_RESULTS_DIR_DEFAULT}/${prefix}_aesthetic.json"

    python -m training.ppo.scripts.score_pick \
        --images "${image_dir}" \
        --pattern "**/*.png" \
        --prompts "${prompt_file}" \
        --weights "${pickscore_weights}" \
        --output-json "${RLEPD_RESULTS_DIR_DEFAULT}/${prefix}_pick.json"

    python -m training.ppo.scripts.score_imagereward \
        --images "${image_dir}" \
        --pattern "**/*.png" \
        --prompts "${prompt_file}" \
        --weights "${imagereward_weights}" \
        --output-json "${RLEPD_RESULTS_DIR_DEFAULT}/${prefix}_imagereward.json"

    python -m training.ppo.scripts.score_mps \
        --images "${image_dir}" \
        --pattern "**/*.png" \
        --prompts "${prompt_file}" \
        --weights "${mps_weights}" \
        --output-json "${RLEPD_RESULTS_DIR_DEFAULT}/${prefix}_mps.json"
}
