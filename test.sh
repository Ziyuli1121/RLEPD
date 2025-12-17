# python fake_train.py \
#   --outdir exps/1 \
#   --num-steps 7 \
#   --num-points 2 \
#   --guidance-rate 4.5 \
#   --schedule-type flowmatch \
#   --backend sd3 \
#   --resolution 1024 \
#   --backend-options '{"model_name_or_path":"stabilityai/stable-diffusion-3-medium-diffusers"}' \
#   --r-base 0.5 \
#   --r-epsilon 0.33

run_sd3_set() {
  local prompt="$1"
  local seeds="${2:-0}"

  python sample_sd3.py --predictor exps/fake-sd3-9-512/network-snapshot-000005.pkl \
    --seeds "$seeds" \
    --outdir training_step_images/1 \
    --prompt "$prompt"

  python sample_sd3.py --predictor exps/fake-sd3-11-512/network-snapshot-000005.pkl \
    --seeds "$seeds" \
    --outdir training_step_images/2 \
    --prompt "$prompt"

  python sample_sd3.py --predictor exps/20251210-005434-sd3_512/export/network-snapshot-export-step001000.pkl \
    --seeds "$seeds" \
    --outdir training_step_images/3 \
    --prompt "$prompt"

  python sample_sd3.py --predictor exps/20251210-005434-sd3_512/export/network-snapshot-export-step005000.pkl \
    --seeds "$seeds" \
    --outdir training_step_images/4 \
    --prompt "$prompt"

  python sample_sd3.py --predictor exps/20251210-005434-sd3_512/export/network-snapshot-export-step009000.pkl \
    --seeds "$seeds" \
    --outdir training_step_images/5 \
    --prompt "$prompt"
}


run_sd3_set 'A photo of a cow left of a stop sign.' '8'
