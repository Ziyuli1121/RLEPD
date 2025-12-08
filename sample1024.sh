# # sd3 baseline


python sample_sd3_baseline.py --sampler edm --resolution 1024 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" --batch 8 \
  --num-steps 10 \
  --outdir ./samples/sd3_edm_20_1024

python sample_sd3_baseline.py --sampler dpm2 --resolution 1024 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" --batch 8 \
  --num-steps 10 \
  --outdir ./samples/sd3_dpm2_20_1024

python sample_sd3_baseline.py --sampler ipndm --resolution 1024  \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" --batch 8 \
  --num-steps 20 --max-order 4 \
  --outdir ./samples/sd3_ipndm4_20_1024

python sample_sd3_baseline.py --sampler sd3 --resolution 1024  \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" --batch 8 \
  --num-steps 20 \
  --outdir ./samples/sd3_default_20_1024_nofinal
####################


# python sample_sd3_baseline.py --sampler edm --resolution 1024 \
#   --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" --batch 8 \
#   --num-steps 14 \
#   --outdir ./samples/sd3_edm_28_1024

python sample_sd3_baseline.py --sampler dpm2 --resolution 1024 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" --batch 8 \
  --num-steps 14 \
  --outdir ./samples/sd3_dpm2_28_1024

python sample_sd3_baseline.py --sampler ipndm --resolution 1024 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" --batch 8 \
  --num-steps 28 --max-order 4 \
  --outdir ./samples/sd3_ipndm4_28_1024

# python sample_sd3_baseline.py --sampler sd3 --resolution 1024 \
#   --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" --batch 8 \
#   --num-steps 28 \
#   --outdir ./samples/sd3_default_28_1024_nofinal
  
# sd3 epd
python -m training.ppo.export_epd_predictor \
  exps/20251206-131339-sd3_1024 \
  --checkpoint checkpoints/policy-step007000.pt

python sample_sd3.py \
  --predictor exps/20251206-131339-sd3_1024/export/network-snapshot-export-step007000.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/sd3_epd_1024_7000

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


score_all_metrics sd3_edm_20_1024
score_all_metrics sd3_dpm2_20_1024
score_all_metrics sd3_ipndm4_20_1024
score_all_metrics sd3_default_20_1024_nofinal

score_all_metrics sd3_dpm2_28_1024
score_all_metrics sd3_ipndm4_28_1024

score_all_metrics sd3_epd_1024_7000
