conda activate epd
cd /work/nvme/betk/zli42/RLEPD

unset RLEPD_PIPELINE_COMMON_SH
source scripts/pipeline_common.sh

score_fid_npz_dir \
  "samples/flux_epd_step002800_coco10k" \
  "src/ms_coco-512x512.npz" \
  "flux_epd_step002800_coco10k" \
  "results/legacy_fid_20260414"

score_fid_npz_dir \
  "samples/flux_baseline_ipndm_24_coco10k" \
  "src/ms_coco-512x512.npz" \
  "flux_baseline_ipndm_24_coco10k" \
  "results/legacy_fid_20260414"

score_fid_npz_dir \
  "samples/flux_baseline_dpm2_12_coco10k" \
  "src/ms_coco-512x512.npz" \
  "flux_baseline_dpm2_12_coco10k" \
  "results/legacy_fid_20260414"