python -m training.ppo.export_epd_predictor \
  exps/20251214-012304-sd3_1024_continue \
  --checkpoint checkpoints/policy-step000800.pt

python -m training.ppo.export_epd_predictor \
  exps/20251214-012304-sd3_1024_continue \
  --checkpoint checkpoints/policy-step001800.pt

python -m training.ppo.export_epd_predictor \
  exps/20251214-012304-sd3_1024_continue \
  --checkpoint checkpoints/policy-step002800.pt

python -m training.ppo.export_epd_predictor \
  exps/20251214-012304-sd3_1024_continue \
  --checkpoint checkpoints/policy-step003800.pt

python -m training.ppo.export_epd_predictor \
  exps/20251214-012304-sd3_1024_continue \
  --checkpoint checkpoints/policy-step004800.pt

python sample_sd3.py \
  --predictor exps/20251214-012304-sd3_1024_continue/export/network-snapshot-export-step000800.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/sd3_1024_8000

python sample_sd3.py \
  --predictor exps/20251214-012304-sd3_1024_continue/export/network-snapshot-export-step001800.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/sd3_1024_9000

python sample_sd3.py \
  --predictor exps/20251214-012304-sd3_1024_continue/export/network-snapshot-export-step002800.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/sd3_1024_10000

python sample_sd3.py \
  --predictor exps/20251214-012304-sd3_1024_continue/export/network-snapshot-export-step003800.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/sd3_1024_11000

python sample_sd3.py \
  --predictor exps/20251214-012304-sd3_1024_continue/export/network-snapshot-export-step004800.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/sd3_1024_12000

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

score_all_metrics sd3_1024_8000
