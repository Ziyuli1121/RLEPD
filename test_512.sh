python -m training.ppo.export_epd_predictor \
  exps/20251210-005434-sd3_512 \
  --checkpoint checkpoints/policy-step011000.pt

python -m training.ppo.export_epd_predictor \
  exps/20251210-005434-sd3_512 \
  --checkpoint checkpoints/policy-step012000.pt

python -m training.ppo.export_epd_predictor \
  exps/20251210-005434-sd3_512 \
  --checkpoint checkpoints/policy-step013000.pt


python sample_sd3.py \
  --predictor exps/20251210-005434-sd3_512/export/network-snapshot-export-step011000.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/sd3_512_11000

python sample_sd3.py \
  --predictor exps/20251210-005434-sd3_512/export/network-snapshot-export-step012000.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/sd3_512_12000

python sample_sd3.py \
  --predictor exps/20251210-005434-sd3_512/export/network-snapshot-export-step013000.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/sd3_512_13000


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


score_all_metrics sd3_512_11000
score_all_metrics sd3_512_12000
score_all_metrics sd3_512_13000
