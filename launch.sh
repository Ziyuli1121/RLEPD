# RL Train

## SD1.5
torchrun --master_port=23123 --nproc_per_node=1 -m training.ppo.launch \
    --config training/ppo/cfgs/sd15_k20.yaml

## SD3-Medium (512x512)
torchrun --master_port=22222 --nproc_per_node=1 -m training.ppo.launch \
    --config training/ppo/cfgs/sd3_512.yaml

## SD3-Medium (1024x1024)
torchrun --master_port=12345 --nproc_per_node=1 -m training.ppo.launch \
    --config training/ppo/cfgs/sd3_1024.yaml

# Sample

## SD1.5
MASTER_PORT=12345 python sample.py \
    --predictor_path exps/20251118-151316-sd15_rl_base/export/network-snapshot-export-step000005.pkl \
    --prompt-file src/prompts/test.txt \
    --seeds "0-999" \
    --batch 16 \
    --outdir samples/sd15

## SD3-Medium
python sample_sd3.py --predictor exps/20251210-011145-sd3_1024/export/network-snapshot-export-step007200.pkl \
  --seeds "0" \
  --outdir output_images \
  --prompt "..."

# Evaluation 
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

score_all_metrics ...