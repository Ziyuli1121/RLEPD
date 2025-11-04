python sample_baseline.py --sampler ipndm \
    --dataset-name ms_coco \
    --prompt-file src/prompts/test.txt \
    --seeds "0-999" --batch 16 \
    --num-steps 51 --schedule-type time_uniform --schedule-rho 1.0 \
    --max-order 3 \
    --outdir ./samples/test_ipndm_nfe50

python sample_baseline.py --sampler ipndm \
    --dataset-name ms_coco \
    --prompt-file src/prompts/test.txt \
    --seeds "0-999" --batch 16 \
    --num-steps 51 --schedule-type discrete --schedule-rho 1.0 \
    --max-order 3 \
    --outdir ./samples/test_ipndm_nfe50_discrete

MASTER_PORT=29600 python sample.py \
    --predictor_path exps/99999-ms_coco-11-20-epd-dpm-1-discrete/network-snapshot-099999.pkl \
    --prompt-file src/prompts/test.txt \
    --seeds "0-9" \
    --batch 16 \
    --outdir ./samples/testtesttest_2

#############################################################################

python -m training.ppo.scripts.score_clip \
    --images samples/test_rl_5450_nfe18 \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/clip \
    --output-json results/test_rl_5450_nfe18_clip.json

python -m training.ppo.scripts.score_hps \
    --images samples/test_rl_5450_nfe18 \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/HPS_v2.1_compressed.pt \
    --output-json results/test_rl_5450_nfe18.json

python -m training.ppo.scripts.score_aesthetic \
    --images samples/test_rl_5450_nfe18 \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/sac+logos+ava1-l14-linearMSE.pth \
    --output-json results/test_rl_5450_nfe18_aesthetic.json

python -m training.ppo.scripts.score_pick \
    --images samples/test_rl_5450_nfe18 \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/PickScore_v1 \
    --output-json results/test_rl_5450_nfe18_pick.json

python -m training.ppo.scripts.score_imagereward \
    --images samples/test_rl_5450_nfe18 \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/ImageReward-v1.0.pt \
    --output-json results/test_rl_5450_nfe18_imagereward.json