# # sd1.5 baseline
# python sample_baseline.py --sampler ddim \
#     --dataset-name ms_coco \
#     --prompt-file src/prompts/test.txt \
#     --seeds "0-99" --batch 16 \
#     --num-steps 10 --schedule-type time_uniform --schedule-rho 1.0 \
#     --outdir ./samples/111

# python sample_baseline.py --sampler edm \
#     --dataset-name ms_coco \
#     --prompt-file src/prompts/test.txt \
#     --seeds "0-99" --batch 16 \
#     --num-steps 5 --schedule-type polynomial --schedule-rho 7.0 \
#     --outdir ./samples/222

# python sample_baseline.py --sampler dpm2 \
#     --dataset-name ms_coco \
#     --prompt-file src/prompts/test.txt \
#     --seeds "0-99" --batch 16 \
#     --num-steps 5 --schedule-type logsnr --schedule-rho 1.0 \
#     --outdir ./samples/333

# python sample_baseline.py --sampler ipndm \
#     --dataset-name ms_coco \
#     --prompt-file src/prompts/test.txt \
#     --seeds "0-99" --batch 16 \
#     --num-steps 10 --schedule-type time_uniform --schedule-rho 1.0 \
#     --max-order 3 \
#     --outdir ./samples/444

# python sample_baseline.py --sampler heun \
#     --dataset-name ms_coco \
#     --prompt-file src/prompts/test.txt \
#     --seeds "0-999" --batch 16 \
#     --num-steps 26 --schedule-type time_uniform --schedule-rho 1.0 \
#     --outdir ./samples/test_heun_nfe50_uni

# python sample_baseline.py --sampler heun \
#     --dataset-name ms_coco \
#     --prompt-file src/prompts/test.txt \
#     --seeds "0-999" --batch 16 \
#     --num-steps 26 --schedule-type polynomial --schedule-rho 1.0 \
#     --outdir ./samples/test_heun_nfe50_poly

# # sd1.5 epd
# MASTER_PORT=29600 python sample.py \
#     --predictor_path exps/20251118-151316-sd15_rl_base/export/network-snapshot-export-step000005.pkl \
#     --prompt-file src/prompts/test.txt \
#     --seeds "0-999" \
#     --batch 16 \
#     --outdir ./samples/ttt

# # sd3 baseline
python sample_sd3_baseline.py --sampler sd3 --resolution 512 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" --batch 8 \
  --num-steps 16 \
  --outdir ./samples/sd3_default_16_512

python sample_sd3_baseline.py --sampler edm --resolution 512 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" --batch 8 \
  --num-steps 9 \
  --outdir ./samples/sd3_edm_flowmatch_16_512

# python sample_sd3_baseline.py --sampler edm --schedule-type polynomial \
#   --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" --batch 8 \
#   --num-steps 8 \
#   --outdir ./samples/sd3_edm_poly

python sample_sd3_baseline.py --sampler dpm2 --resolution 512 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" --batch 8 \
  --num-steps 9 \
  --outdir ./samples/sd3_dpm2_flowmatch_16_512

# python sample_sd3_baseline.py --sampler dpm2 --schedule-type logsnr \
#   --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" --batch 8 \
#   --num-steps 8 \
#   --outdir ./samples/sd3_dpm2_logsnr

python sample_sd3_baseline.py --sampler ipndm --resolution 512 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" --batch 8 \
  --num-steps 16 \
  --outdir ./samples/sd3_ipndm_flowmatch_16_512

# python sample_sd3_baseline.py --sampler ipndm --schedule-type time_uniform \
#   --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" --batch 8 \
#   --num-steps 16 \
#   --outdir ./samples/sd3_ipndm_uniform

# # sd3 epd
# python sample_sd3.py \
#   --predictor exps/fake-sd3-9/network-snapshot-000005.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --max-batch-size 4 \
#   --outdir samples/sd3_epd_9

# python sample_sd3.py \
#   --predictor exps/20251123-215008-sd3_smoke/export/network-snapshot-export-step002900.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --max-batch-size 4 \
#   --outdir samples/sd3_epd_9_2900

python sample_sd3.py \
  --predictor exps/20251201-145511-sd3_512/export/network-snapshot-export-step002000.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 1 \
  --outdir samples/sd3_epd_9_512_2000
  
# # sd3.5 baseline

# python sample_sd3_baseline.py --sampler sd3 \
#   --model-id "stabilityai/stable-diffusion-3.5-medium" \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" --batch 16 \
#   --num-steps 28 \
#   --outdir ./samples/sd35_baseline

# # sd3.5 epd

# python sample_sd3.py \
#   --predictor exps/fake-sd35-15/network-snapshot-000005.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --max-batch-size 4 \
#   --outdir samples/sd35_epd_15



score_all_metrics() {
    local name="$1"
    if [ -z "$name" ]; then
        echo "Usage: score_all_metrics <images_subdir_under_samples>"
        return 1
    fi

    local image_dir="samples/${name}"
    local prefix="${name}"

    mkdir -p results

    python -m training.ppo.scripts.score_clip \
        --images "${image_dir}" \
        --pattern "**/*.png" \
        --prompts src/prompts/test.txt \
        --weights weights/clip \
        --output-json "results/${prefix}_clip.json"

    python -m training.ppo.scripts.score_hps \
        --images "${image_dir}" \
        --pattern "**/*.png" \
        --prompts src/prompts/test.txt \
        --weights weights/HPS_v2.1_compressed.pt \
        --output-json "results/${prefix}_hps.json"

    python -m training.ppo.scripts.score_aesthetic \
        --images "${image_dir}" \
        --pattern "**/*.png" \
        --prompts src/prompts/test.txt \
        --weights weights/sac+logos+ava1-l14-linearMSE.pth \
        --output-json "results/${prefix}_aesthetic.json"

    python -m training.ppo.scripts.score_pick \
        --images "${image_dir}" \
        --pattern "**/*.png" \
        --prompts src/prompts/test.txt \
        --weights weights/PickScore_v1 \
        --output-json "results/${prefix}_pick.json"

    python -m training.ppo.scripts.score_imagereward \
        --images "${image_dir}" \
        --pattern "**/*.png" \
        --prompts src/prompts/test.txt \
        --weights weights/ImageReward-v1.0.pt \
        --output-json "results/${prefix}_imagereward.json"

    python -m training.ppo.scripts.score_mps \
        --images "${image_dir}" \
        --pattern "**/*.png" \
        --prompts src/prompts/test.txt \
        --weights weights/MPS_overall_checkpoint.pth \
        --output-json "results/${prefix}_mps.json"
}

# Change the name below to score a different folder under samples/.
sleep 3
score_all_metrics sd3_epd_9_512_2000


# visualize dirichlet
# python visualize_dirichlet.py \
#     --checkpoint /work/nvme/betk/zli42/RLEPD/exps/20251114-202156-sd15_rl_base/export/network-snapshot-export-step003500.pkl \
#     --output dirichlet_heatmap.png \
#     --surface-step 5 \
#     --surface-target position \
#     --surface-output beta_surface_step5.png \
#     --concentration 10