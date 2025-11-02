#!/usr/bin/env bash

# Stage 9+ 常用命令备忘脚本（本脚本不再自动执行命令，只列示参考）

# 1. 训练（可添加额外 override）
python -m training.ppo.launch \
    --config training/ppo/cfgs/sd15_base.yaml

# 2. 导出策略均值为 EPD predictor
python -m training.ppo.export_epd_predictor \
    exps/20251030-235041-sd15_rl_base \
    --checkpoint checkpoints/policy-step005000.pt

# 3. 使用导出的 predictor 生成图像

# 不写prompt file，默认是30k mscoco prompt
python sample.py \
    --predictor_path exps/20251030-235041-sd15_rl_base/export/network-snapshot-export-step001700.pkl \
    --batch 1 --seeds "0-10" \
    --outdir ./samples/rl_new

# epd （distillation / RL）
python sample.py \
    --predictor_path exps/20251030-235041-sd15_rl_base/export/network-snapshot-export-step005000.pkl \
    --prompt-file src/prompts/test.txt \
    --seeds "0-99" \
    --batch 16 \
    --outdir ./samples/test_rl_5000

python sample_baseline.py --sampler ddim \
    --dataset-name ms_coco \
    --prompt-file src/prompts/test.txt \
    --seeds "0-99" --batch 16 \
    --num-steps 36 --ddim-steps 36 --ddim-eta 0.0 \
    --outdir ./samples/test_ddim_newddim_36

python sample_baseline.py --sampler ddim \
    --dataset-name ms_coco \
    --prompt-file src/prompts/test.txt \
    --seeds "0-999" --batch 16 \
    --num-steps 51 --ddim-steps 51 --ddim-eta 0.0 \
    --outdir ./samples/test_ddim_51

python sample_baseline.py --sampler edm \
    --dataset-name ms_coco \
    --prompt-file src/prompts/test.txt \
    --seeds "0-99" --batch 16 \
    --num-steps 18 --edm-s-churn 0 --edm-s-noise 1 \
    --outdir ./samples/test_edm

python sample_baseline.py --sampler dpm \
    --dataset-name ms_coco \
    --prompt-file src/prompts/test.txt \
    --seeds "0-99" --batch 16 \
    --num-steps 9 --inner-steps 2 --solver-r 0.5 \
    --outdir ./samples/test_dpm

python sample_baseline.py --sampler ipndm \
    --dataset-name ms_coco \
    --prompt-file src/prompts/test.txt \
    --seeds 0-99 --batch 16 \
    --num-steps 36 --max-order 3 \
    --outdir ./samples/test_ipndm

# 4. 评估生成图像的 HPS 分数
python -m training.ppo.scripts.score_hps \
    --images samples/test_rl \
    --pattern "**/*.png" \
    --prompts src/prompts/test100.txt \
    --weights weights/HPS_v2.1_compressed.pt

# 0.24169921875


python -m training.ppo.scripts.score_hps \
    --images samples/test_distillation \
    --pattern "**/*.png" \
    --prompts src/prompts/test100.txt \
    --weights weights/HPS_v2.1_compressed.pt

# 0.2410888671875

python -m training.ppo.scripts.score_hps \
    --images samples/test_ddim \
    --pattern "**/*.png" \
    --prompts src/prompts/test100.txt \
    --weights weights/HPS_v2.1_compressed.pt

# 0.24365234375

python -m training.ppo.scripts.score_hps \
    --images samples/test_ddim_few \
    --pattern "**/*.png" \
    --prompts src/prompts/test100.txt \
    --weights weights/HPS_v2.1_compressed.pt

# 0.2403564453125

python -m training.ppo.scripts.score_hps \
    --images samples/test_edm \
    --pattern "**/*.png" \
    --prompts src/prompts/test100.txt \
    --weights weights/HPS_v2.1_compressed.pt

# 0.2427978515625

python -m training.ppo.scripts.score_hps \
    --images samples/test_dpm \
    --pattern "**/*.png" \
    --prompts src/prompts/test100.txt \
    --weights weights/HPS_v2.1_compressed.pt

# 0.2381591796875

python -m training.ppo.scripts.score_hps \
    --images samples/test_rl_2800 \
    --pattern "**/*.png" \
    --prompts src/prompts/test100.txt \
    --weights weights/HPS_v2.1_compressed.pt

# 0.2420654296875

python -m training.ppo.scripts.score_hps \
    --images samples/test_ddim_51 \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/HPS_v2.1_compressed.pt

# 0.2464599609375

python -m training.ppo.scripts.score_hps \
    --images samples/test_rl_5000 \
    --pattern "**/*.png" \
    --prompts src/prompts/test100.txt \
    --weights weights/HPS_v2.1_compressed.pt

# 0.243408203125

python -m training.ppo.scripts.score_hps \
    --images samples/test_ddim_newddim_36 \
    --pattern "**/*.png" \
    --prompts src/prompts/test100.txt \
    --weights weights/HPS_v2.1_compressed.pt

# 0.24267578125