# python -m training.ppo.export_epd_predictor \
#   exps/20251214-134452-sd15_k5 \
#   --checkpoint checkpoints/policy-step001000.pt

# python -m training.ppo.export_epd_predictor \
#   exps/20251214-134452-sd15_k5 \
#   --checkpoint checkpoints/policy-step002000.pt

# python -m training.ppo.export_epd_predictor \
#   exps/20251214-134452-sd15_k5 \
#   --checkpoint checkpoints/policy-step003000.pt

# python -m training.ppo.export_epd_predictor \
#   exps/20251214-134452-sd15_k5 \
#   --checkpoint checkpoints/policy-step004000.pt

# python -m training.ppo.export_epd_predictor \
#   exps/20251214-134452-sd15_k5 \
#   --checkpoint checkpoints/policy-step005000.pt

# python -m training.ppo.export_epd_predictor \
#   exps/20251214-134452-sd15_k5 \
#   --checkpoint checkpoints/policy-step006000.pt

# python -m training.ppo.export_epd_predictor \
#   exps/20251214-134452-sd15_k5 \
#   --checkpoint checkpoints/policy-step007000.pt

# python -m training.ppo.export_epd_predictor \
#   exps/20251214-134452-sd15_k5 \
#   --checkpoint checkpoints/policy-step008000.pt

# python -m training.ppo.export_epd_predictor \
#   exps/20251214-134452-sd15_k5 \
#   --checkpoint checkpoints/policy-step009000.pt

python -m training.ppo.export_epd_predictor \
  exps/20251214-134452-sd15_k5 \
  --checkpoint checkpoints/policy-step010000.pt

###########################################################

# python -m training.ppo.export_epd_predictor \
#   exps/20251214-135518-sd15_k20 \
#   --checkpoint checkpoints/policy-step001000.pt

# python -m training.ppo.export_epd_predictor \
#   exps/20251214-135518-sd15_k20 \
#   --checkpoint checkpoints/policy-step002000.pt

# python -m training.ppo.export_epd_predictor \
#   exps/20251214-135518-sd15_k20 \
#   --checkpoint checkpoints/policy-step003000.pt

# python -m training.ppo.export_epd_predictor \
#   exps/20251214-135518-sd15_k20 \
#   --checkpoint checkpoints/policy-step004000.pt

# python -m training.ppo.export_epd_predictor \
#   exps/20251214-135518-sd15_k20 \
#   --checkpoint checkpoints/policy-step005000.pt

# python -m training.ppo.export_epd_predictor \
#   exps/20251214-135518-sd15_k20 \
#   --checkpoint checkpoints/policy-step006000.pt

# python -m training.ppo.export_epd_predictor \
#   exps/20251214-135518-sd15_k20 \
#   --checkpoint checkpoints/policy-step007000.pt

# python -m training.ppo.export_epd_predictor \
#   exps/20251214-135518-sd15_k20 \
#   --checkpoint checkpoints/policy-step008000.pt

# python -m training.ppo.export_epd_predictor \
#   exps/20251214-135518-sd15_k20 \
#   --checkpoint checkpoints/policy-step009000.pt

python -m training.ppo.export_epd_predictor \
  exps/20251214-135518-sd15_k20 \
  --checkpoint checkpoints/policy-step010000.pt

#############################################################

# python -m training.ppo.export_epd_predictor \
#   exps/20251214-135518-sd15_k50 \
#   --checkpoint checkpoints/policy-step001000.pt

# python -m training.ppo.export_epd_predictor \
#   exps/20251214-135518-sd15_k50 \
#   --checkpoint checkpoints/policy-step002000.pt

# python -m training.ppo.export_epd_predictor \
#   exps/20251214-135518-sd15_k50 \
#   --checkpoint checkpoints/policy-step003000.pt

# python -m training.ppo.export_epd_predictor \
#   exps/20251214-135518-sd15_k50 \
#   --checkpoint checkpoints/policy-step004000.pt

# python -m training.ppo.export_epd_predictor \
#   exps/20251214-135518-sd15_k50 \
#   --checkpoint checkpoints/policy-step005000.pt

# python -m training.ppo.export_epd_predictor \
#   exps/20251214-135518-sd15_k50 \
#   --checkpoint checkpoints/policy-step006000.pt

# python -m training.ppo.export_epd_predictor \
#   exps/20251214-135518-sd15_k50 \
#   --checkpoint checkpoints/policy-step007000.pt

# python -m training.ppo.export_epd_predictor \
#   exps/20251214-135518-sd15_k50 \
#   --checkpoint checkpoints/policy-step008000.pt

# python -m training.ppo.export_epd_predictor \
#   exps/20251214-135518-sd15_k50 \
#   --checkpoint checkpoints/policy-step009000.pt

python -m training.ppo.export_epd_predictor \
  exps/20251214-135518-sd15_k50 \
  --checkpoint checkpoints/policy-step010000.pt

#############################################################

# MASTER_PORT=22222 python sample.py \
#   --predictor_path exps/20251214-134452-sd15_k5/export/network-snapshot-export-step001000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k5_1000

# MASTER_PORT=33333 python sample.py \
#   --predictor_path exps/20251214-134452-sd15_k5/export/network-snapshot-export-step002000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k5_2000

# MASTER_PORT=11111 python sample.py \
#   --predictor_path exps/20251214-134452-sd15_k5/export/network-snapshot-export-step003000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k5_3000

# MASTER_PORT=22222 python sample.py \
#   --predictor_path exps/20251214-134452-sd15_k5/export/network-snapshot-export-step004000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k5_4000

# MASTER_PORT=33333 python sample.py \
#   --predictor_path exps/20251214-134452-sd15_k5/export/network-snapshot-export-step005000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k5_5000

# MASTER_PORT=11111 python sample.py \
#   --predictor_path exps/20251214-134452-sd15_k5/export/network-snapshot-export-step006000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k5_6000

# MASTER_PORT=22222 python sample.py \
#   --predictor_path exps/20251214-134452-sd15_k5/export/network-snapshot-export-step007000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k5_7000

# MASTER_PORT=33333 python sample.py \
#   --predictor_path exps/20251214-134452-sd15_k5/export/network-snapshot-export-step008000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k5_8000

# MASTER_PORT=11111 python sample.py \
#   --predictor_path exps/20251214-134452-sd15_k5/export/network-snapshot-export-step009000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k5_9000

# MASTER_PORT=44444 python sample.py \
#   --predictor_path exps/20251214-134452-sd15_k5/export/network-snapshot-export-step010000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k5_10000

##############################################################

# MASTER_PORT=22222 python sample.py \
#   --predictor_path exps/20251214-135518-sd15_k20/export/network-snapshot-export-step001000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k20_1000

# MASTER_PORT=33333 python sample.py \
#   --predictor_path exps/20251214-135518-sd15_k20/export/network-snapshot-export-step002000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k20_2000

# MASTER_PORT=11111 python sample.py \
#   --predictor_path exps/20251214-135518-sd15_k20/export/network-snapshot-export-step003000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k20_3000

# MASTER_PORT=22222 python sample.py \
#   --predictor_path exps/20251214-135518-sd15_k20/export/network-snapshot-export-step004000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k20_4000

# MASTER_PORT=33333 python sample.py \
#   --predictor_path exps/20251214-135518-sd15_k20/export/network-snapshot-export-step005000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k20_5000

# MASTER_PORT=11111 python sample.py \
#   --predictor_path exps/20251214-135518-sd15_k20/export/network-snapshot-export-step006000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k20_6000

# MASTER_PORT=22222 python sample.py \
#   --predictor_path exps/20251214-135518-sd15_k20/export/network-snapshot-export-step007000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k20_7000

# MASTER_PORT=33333 python sample.py \
#   --predictor_path exps/20251214-135518-sd15_k20/export/network-snapshot-export-step008000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k20_8000

# MASTER_PORT=11111 python sample.py \
#   --predictor_path exps/20251214-135518-sd15_k20/export/network-snapshot-export-step009000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k20_9000

MASTER_PORT=55555 python sample.py \
  --predictor_path exps/20251214-135518-sd15_k20/export/network-snapshot-export-step010000.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --batch 16 \
  --outdir samples/sd15_k20_10000

#########################################################

# MASTER_PORT=22222 python sample.py \
#   --predictor_path exps/20251214-135518-sd15_k50/export/network-snapshot-export-step001000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k50_1000

# MASTER_PORT=33333 python sample.py \
#   --predictor_path exps/20251214-135518-sd15_k50/export/network-snapshot-export-step002000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k50_2000

# MASTER_PORT=11111 python sample.py \
#   --predictor_path exps/20251214-135518-sd15_k50/export/network-snapshot-export-step003000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k50_3000

# MASTER_PORT=22222 python sample.py \
#   --predictor_path exps/20251214-135518-sd15_k50/export/network-snapshot-export-step004000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k50_4000

# MASTER_PORT=33333 python sample.py \
#   --predictor_path exps/20251214-135518-sd15_k50/export/network-snapshot-export-step005000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k50_5000

# MASTER_PORT=11111 python sample.py \
#   --predictor_path exps/20251214-135518-sd15_k50/export/network-snapshot-export-step006000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k50_6000

# MASTER_PORT=22222 python sample.py \
#   --predictor_path exps/20251214-135518-sd15_k50/export/network-snapshot-export-step007000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k50_7000

# MASTER_PORT=33333 python sample.py \
#   --predictor_path exps/20251214-135518-sd15_k50/export/network-snapshot-export-step008000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k50_8000

# MASTER_PORT=11111 python sample.py \
#   --predictor_path exps/20251214-135518-sd15_k50/export/network-snapshot-export-step009000.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --batch 16 \
#   --outdir samples/sd15_k50_9000

MASTER_PORT=11888 python sample.py \
  --predictor_path exps/20251214-135518-sd15_k50/export/network-snapshot-export-step010000.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --batch 16 \
  --outdir samples/sd15_k50_10000









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

# score_all_metrics sd15_k5_1000
# score_all_metrics sd15_k5_2000
# score_all_metrics sd15_k5_3000
# score_all_metrics sd15_k5_4000
# score_all_metrics sd15_k5_5000
# score_all_metrics sd15_k5_6000
# score_all_metrics sd15_k5_7000
# score_all_metrics sd15_k5_8000
# score_all_metrics sd15_k5_9000
score_all_metrics sd15_k5_10000

# score_all_metrics sd15_k20_1000
# score_all_metrics sd15_k20_2000
# score_all_metrics sd15_k20_3000
# score_all_metrics sd15_k20_4000
# score_all_metrics sd15_k20_5000
# score_all_metrics sd15_k20_6000
# score_all_metrics sd15_k20_7000
# score_all_metrics sd15_k20_8000
# score_all_metrics sd15_k20_9000
score_all_metrics sd15_k20_10000

# score_all_metrics sd15_k50_1000
# score_all_metrics sd15_k50_2000
score_all_metrics sd15_k50_3000
score_all_metrics sd15_k50_4000
score_all_metrics sd15_k50_5000
score_all_metrics sd15_k50_6000
score_all_metrics sd15_k50_7000
score_all_metrics sd15_k50_8000
score_all_metrics sd15_k50_9000
score_all_metrics sd15_k50_10000