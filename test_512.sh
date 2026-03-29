python -m training.ppo.export_epd_predictor \
  exps/20251219-032038-sd3_512_new \
  --checkpoint checkpoints/policy-step015000.pt

python -m training.ppo.export_epd_predictor \
  exps/20251219-032038-sd3_512_new \
  --checkpoint checkpoints/policy-step016000.pt

python -m training.ppo.export_epd_predictor \
  exps/20251219-032038-sd3_512_new \
  --checkpoint checkpoints/policy-step017000.pt

python -m training.ppo.export_epd_predictor \
  exps/20251219-032038-sd3_512_new \
  --checkpoint checkpoints/policy-step018000.pt

#############################################################

python sample_sd3.py \
  --predictor exps/20251219-032038-sd3_512_new/export/network-snapshot-export-step015000.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/512_15000

python sample_sd3.py \
  --predictor exps/20251219-032038-sd3_512_new/export/network-snapshot-export-step016000.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/512_16000

python sample_sd3.py \
  --predictor exps/20251219-032038-sd3_512_new/export/network-snapshot-export-step017000.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/512_17000

python sample_sd3.py \
  --predictor exps/20251219-032038-sd3_512_new/export/network-snapshot-export-step018000.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/512_18000

##############################################################

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

score_all_metrics 512_15000
score_all_metrics 512_16000
score_all_metrics 512_17000
score_all_metrics 512_18000