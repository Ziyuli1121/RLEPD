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

# # sd1.5 epd
# MASTER_PORT=29600 python sample.py \
#     --predictor_path exps/20251118-151316-sd15_rl_base/export/network-snapshot-export-step000005.pkl \
#     --prompt-file src/prompts/test.txt \
#     --seeds "0-999" \
#     --batch 16 \
#     --outdir ./samples/ttt

# # sd3 baseline
# python sample_sd3_baseline.py --sampler sd3 \
#   --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" --batch 16 \
#   --num-steps 28 \
#   --outdir ./samples/sd3_baseline

# # sd3 epd
# python sample_sd3.py \
#   --predictor exps/fake-sd3-15/network-snapshot-000005.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --max-batch-size 4 \
#   --outdir samples/sd3_epd_15

# python sample_sd3.py \
#   --predictor exps/fake-sd3-9/network-snapshot-000005.pkl \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" \
#   --max-batch-size 4 \
#   --outdir samples/sd3_epd_9
  
# # sd3.5 baseline

# python sample_sd3_baseline.py --sampler sd3 \
#   --model-id "stabilityai/stable-diffusion-3.5-medium" \
#   --prompt-file src/prompts/test.txt \
#   --seeds "0-999" --batch 16 \
#   --num-steps 28 \
#   --outdir ./samples/sd35_baseline

# # sd3.5 epd

python sample_sd3.py \
  --predictor exps/fake-sd35-15/network-snapshot-000005.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/sd35_epd_15

python sample_sd3.py \
  --predictor exps/fake-sd35-9/network-snapshot-000005.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --max-batch-size 4 \
  --outdir samples/sd35_epd_9




# evaluation
python -m training.ppo.scripts.score_clip \
    --images samples/sd35_baseline \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/clip \
    --output-json results/sd35_baseline_clip.json

python -m training.ppo.scripts.score_hps \
    --images samples/sd35_baseline \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/HPS_v2.1_compressed.pt \
    --output-json results/sd35_baseline_hps.json

python -m training.ppo.scripts.score_aesthetic \
    --images samples/sd35_baseline \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/sac+logos+ava1-l14-linearMSE.pth \
    --output-json results/sd35_baseline_aesthetic.json

python -m training.ppo.scripts.score_pick \
    --images samples/sd35_baseline \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/PickScore_v1 \
    --output-json results/sd35_baseline_pick.json

python -m training.ppo.scripts.score_imagereward \
    --images samples/sd35_baseline \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/ImageReward-v1.0.pt \
    --output-json results/sd35_baseline_imagereward.json

python -m training.ppo.scripts.score_mps \
    --images samples/sd35_baseline \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/MPS_overall_checkpoint.pth \
    --output-json results/sd35_baseline_mps.json




# visualize dirichlet
# python visualize_dirichlet.py \
#     --checkpoint /work/nvme/betk/zli42/RLEPD/exps/20251114-202156-sd15_rl_base/export/network-snapshot-export-step003500.pkl \
#     --output dirichlet_heatmap.png \
#     --surface-step 5 \
#     --surface-target position \
#     --surface-output beta_surface_step5.png \
#     --concentration 10