# # sd3 baseline


# python sample_sd3_baseline.py --sampler edm --resolution 512 \
#   --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" --batch 8 \
#   --num-steps 8 \
#   --outdir ./samples/sd3_edm_flowmatch_16_512

# python sample_sd3_baseline.py --sampler dpm2 --resolution 512 \
#   --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" --batch 8 \
#   --num-steps 9 \
#   --outdir ./samples/sd3_dpm2_flowmatch_16_512

# python sample_sd3_baseline.py --sampler ipndm --resolution 512 \
#   --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" --batch 8 \
#   --num-steps 17 --max-order 3 \
#   --outdir ./samples/sd3_ipndm_flowmatch_16_512

# python sample_sd3_baseline.py --sampler sd3 --resolution 512 \
#   --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" --batch 8 \
#   --num-steps 16 \
#   --outdir ./samples/sd3_default_16_512
#####################


# python sample_sd3_baseline.py --sampler edm --resolution 512 \
#   --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" --batch 8 \
#   --num-steps 14 \
#   --outdir ./samples/sd3_edm_flowmatch_28_512

# python sample_sd3_baseline.py --sampler dpm2 --resolution 512 \
#   --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" --batch 8 \
#   --num-steps 15 \
#   --outdir ./samples/sd3_dpm2_flowmatch_28_512

# python sample_sd3_baseline.py --sampler ipndm --resolution 512 \
#   --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" --batch 8 \
#   --num-steps 29 --max-order 3 \
#   --outdir ./samples/sd3_ipndm_flowmatch_28_512

# python sample_sd3_baseline.py --sampler sd3 --resolution 512 \
#   --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" --batch 8 \
#   --num-steps 28 \
#   --outdir ./samples/sd3_default_28_512
  
# # sd3 epd

# python sample_sd3.py \
#   --predictor exps/20251201-201759-sd3_512/export/network-snapshot-export-step002000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --max-batch-size 4 \
#   --outdir samples/sd3_epd_9_512_2000

python sample_sd3.py \
  --predictor exps/20251201-201759-sd3_512/export/network-snapshot-export-step000500.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/sd3_epd_9_512_500

python sample_sd3.py \
  --predictor exps/20251201-201759-sd3_512/export/network-snapshot-export-step001000.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/sd3_epd_9_512_1000
  
python sample_sd3.py \
  --predictor exps/20251201-201759-sd3_512/export/network-snapshot-export-step001500.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/sd3_epd_9_512_1500

python sample_sd3.py \
  --predictor exps/20251201-201759-sd3_512/export/network-snapshot-export-step002450.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/sd3_epd_9_512_2450

####################

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


# score_all_metrics sd3_default_16_512
# score_all_metrics sd3_edm_flowmatch_16_512
# score_all_metrics sd3_dpm2_flowmatch_16_512
# score_all_metrics sd3_ipndm_flowmatch_16_512
# score_all_metrics sd3_epd_9_512_2000
# score_all_metrics sd3_default_28_512
# score_all_metrics sd3_edm_flowmatch_28_512
# score_all_metrics sd3_dpm2_flowmatch_28_512
# score_all_metrics sd3_ipndm_flowmatch_28_512

score_all_metrics sd3_epd_9_512_500
score_all_metrics sd3_epd_9_512_1000
score_all_metrics sd3_epd_9_512_1500
score_all_metrics sd3_epd_9_512_2450
