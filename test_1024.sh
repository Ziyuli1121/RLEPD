python -m training.ppo.export_epd_predictor \
  exps/20251219-032043-sd3_1024_continue \
  --checkpoint checkpoints/policy-step001000.pt

python -m training.ppo.export_epd_predictor \
  exps/20251219-032043-sd3_1024_continue \
  --checkpoint checkpoints/policy-step002000.pt

python -m training.ppo.export_epd_predictor \
  exps/20251219-032043-sd3_1024_continue \
  --checkpoint checkpoints/policy-step003000.pt

python -m training.ppo.export_epd_predictor \
  exps/20251219-032043-sd3_1024_continue \
  --checkpoint checkpoints/policy-step004000.pt

python -m training.ppo.export_epd_predictor \
  exps/20251219-032043-sd3_1024_continue \
  --checkpoint checkpoints/policy-step005000.pt

python -m training.ppo.export_epd_predictor \
  exps/20251219-032043-sd3_1024_continue \
  --checkpoint checkpoints/policy-step006000.pt

python -m training.ppo.export_epd_predictor \
  exps/20251219-032043-sd3_1024_continue \
  --checkpoint checkpoints/policy-step007000.pt

python -m training.ppo.export_epd_predictor \
  exps/20251219-032043-sd3_1024_continue \
  --checkpoint checkpoints/policy-step008000.pt

#############################################################

python sample_sd3.py \
  --predictor exps/20251219-032043-sd3_1024_continue/export/network-snapshot-export-step001000.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/1024_1000

python sample_sd3.py \
  --predictor exps/20251219-032043-sd3_1024_continue/export/network-snapshot-export-step002000.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/1024_2000

python sample_sd3.py \
  --predictor exps/20251219-032043-sd3_1024_continue/export/network-snapshot-export-step003000.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/1024_3000

python sample_sd3.py \
  --predictor exps/20251219-032043-sd3_1024_continue/export/network-snapshot-export-step004000.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/1024_4000

python sample_sd3.py \
  --predictor exps/20251219-032043-sd3_1024_continue/export/network-snapshot-export-step005000.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/1024_5000

python sample_sd3.py \
  --predictor exps/20251219-032043-sd3_1024_continue/export/network-snapshot-export-step006000.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/1024_6000

python sample_sd3.py \
  --predictor exps/20251219-032043-sd3_1024_continue/export/network-snapshot-export-step007000.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/1024_7000

python sample_sd3.py \
  --predictor exps/20251219-032043-sd3_1024_continue/export/network-snapshot-export-step008000.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/1024_8000
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

score_all_metrics 1024_1000
score_all_metrics 1024_2000
score_all_metrics 1024_3000
score_all_metrics 1024_4000
score_all_metrics 1024_5000
score_all_metrics 1024_6000
score_all_metrics 1024_7000
score_all_metrics 1024_8000