# python sample_baseline.py --sampler ddim \
#     --dataset-name ms_coco \
#     --prompt-file src/prompts/MS-COCO_val2014_30k_captions.csv \
#     --seeds "0-29999" --batch 16 \
#     --num-steps 1 --schedule-type time_uniform --schedule-rho 1.0 \
#     --outdir ./samples/test_tokenizer_mscoco

# python sample_baseline.py --sampler ddim \
#     --dataset-name ms_coco \
#     --prompt-file src/prompts/train_ori.csv \
#     --seeds "0-25431" --batch 16 \
#     --num-steps 1 --schedule-type time_uniform --schedule-rho 1.0 \
#     --outdir ./samples/test_tokenizer

# python sample_baseline.py --sampler ddim \
#     --dataset-name ms_coco \
#     --prompt-file src/prompts/train.csv \
#     --seeds "0-24419" --batch 16 \
#     --num-steps 1 --schedule-type time_uniform --schedule-rho 1.0 \
#     --outdir ./samples/test_modified_data

#############################################################################


# python sample_baseline.py --sampler ddim \
#     --dataset-name ms_coco \
#     --prompt-file src/prompts/test.txt \
#     --seeds "0-999" --batch 16 \
#     --num-steps 37 --schedule-type time_uniform --schedule-rho 1.0 \
#     --outdir ./samples/test_ddim_nfe36


# python sample_baseline.py --sampler ddim \
#     --dataset-name ms_coco \
#     --prompt-file src/prompts/test.txt \
#     --seeds "0-999" --batch 16 \
#     --num-steps 37 --schedule-type discrete --schedule-rho 1.0 \
#     --outdir ./samples/test_ddim_nfe36_discrete

# python sample_baseline.py --sampler ipndm \
#     --dataset-name ms_coco \
#     --prompt-file src/prompts/test.txt \
#     --seeds "0-999" --batch 16 \
#     --num-steps 37 --schedule-type time_uniform --schedule-rho 1.0 \
#     --max-order 3 \
#     --outdir ./samples/test_ipndm_nfe36

# python sample_baseline.py --sampler ipndm \
#     --dataset-name ms_coco \
#     --prompt-file src/prompts/test.txt \
#     --seeds "0-999" --batch 16 \
#     --num-steps 37 --schedule-type discrete --schedule-rho 1.0 \
#     --max-order 3 \
#     --outdir ./samples/test_ipndm_nfe36_discrete

# python sample_baseline.py --sampler dpm2 \
#     --dataset-name ms_coco \
#     --prompt-file src/prompts/test.txt \
#     --seeds "0-999" --batch 16 \
#     --num-steps 19 --schedule-type logsnr --schedule-rho 1.0 \
#     --outdir ./samples/test_dpm2_nfe36

# python sample_baseline.py --sampler dpm2 \
#     --dataset-name ms_coco \
#     --prompt-file src/prompts/test.txt \
#     --seeds "0-999" --batch 16 \
#     --num-steps 19 --schedule-type discrete --schedule-rho 1.0 \
#     --outdir ./samples/test_dpm2_nfe36_discrete

# python sample_baseline.py --sampler edm \
#     --dataset-name ms_coco \
#     --prompt-file src/prompts/test.txt \
#     --seeds "0-999" --batch 16 \
#     --num-steps 19 --schedule-type polynomial --schedule-rho 7.0 \
#     --outdir ./samples/test_edm_nfe36




MASTER_PORT=29600 python sample.py \
    --predictor_path exps/20251115-110803-sd15_rl_base/export/network-snapshot-export-step008000.pkl \
    --prompt-file src/prompts/test.txt \
    --seeds "0-999" \
    --batch 16 \
    --outdir ./samples/123123

#############################################################################

python -m training.ppo.scripts.score_clip \
    --images samples/20251115-110803-sd15_rl_base-step008000 \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/clip \
    --output-json results/20251115-110803-sd15_rl_base-step008000_clip.json

python -m training.ppo.scripts.score_hps \
    --images samples/20251115-110803-sd15_rl_base-step008000 \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/HPS_v2.1_compressed.pt \
    --output-json results/20251115-110803-sd15_rl_base-step008000_hps.json

python -m training.ppo.scripts.score_aesthetic \
    --images samples/20251115-110803-sd15_rl_base-step008000 \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/sac+logos+ava1-l14-linearMSE.pth \
    --output-json results/20251115-110803-sd15_rl_base-step008000_aesthetic.json

python -m training.ppo.scripts.score_pick \
    --images samples/20251115-110803-sd15_rl_base-step008000 \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/PickScore_v1 \
    --output-json results/20251115-110803-sd15_rl_base-step008000_pick.json

python -m training.ppo.scripts.score_imagereward \
    --images samples/20251115-110803-sd15_rl_base-step008000 \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/ImageReward-v1.0.pt \
    --output-json results/20251115-110803-sd15_rl_base-step008000_imagereward.json

python -m training.ppo.scripts.score_mps \
    --images samples/mixed_reward_8rl_1800 \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/MPS_overall_checkpoint.pth \
    --output-json results/mixed_reward_8rl_1800_mps.json

