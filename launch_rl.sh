#!/usr/bin/env bash

# Stage 9+ 常用命令备忘脚本（本脚本不再自动执行命令，只列示参考）

# 1. 训练（可添加额外 override）



# 伪训练（设置初始值）
python fake_train.py \
  --num-steps 11 \
  --num-points 2 \
  --outdir exps/99999-ms_coco-11-20-epd-dpm-1-discrete \
  --snapshot-step 99999 \
  --run-dir ./exps/99999-ms_coco-11-20-epd-dpm-1-discrete


# sd15_base.yaml提供PPO训练参数

python -m training.ppo.launch \
    --config training/ppo/cfgs/sd15_base.yaml

torchrun --nproc_per_node=1 -m training.ppo.launch \
    --config training/ppo/cfgs/sd15_base.yaml

#####################################################################################################
torchrun --master_port=29505 --nproc_per_node=1 -m training.ppo.launch \
    --config training/ppo/cfgs/sd15_parallel.yaml


torchrun --master_port=29505 --nproc_per_node=8 -m training.ppo.launch \
    --config training/ppo/cfgs/sd15_parallel.yaml
#####################################################################################################

# 2. 导出策略均值为 EPD predictor
python -m training.ppo.export_epd_predictor \
    exps/20251104-013538-sd15_rl_base \
    --checkpoint checkpoints/policy-step000050.pt

# 3. 使用导出的 predictor 生成图像

# 不写prompt file，默认是30k mscoco prompt
python sample.py \
    --predictor_path exps/20251030-235041-sd15_rl_base/export/network-snapshot-export-step001700.pkl \
    --batch 1 --seeds "0-10" \
    --outdir ./samples/rl_new

# epd （distillation / RL）
MASTER_PORT=29600 python sample.py \
    --predictor_path exps/20251104-013538-sd15_rl_base/export/network-snapshot-export-step000050.pkl \
    --prompt-file src/prompts/test.txt \
    --seeds "0-99" \
    --batch 16 \
    --outdir ./samples/nnnnnnn


# 4. 评估生成图像的 HPS 分数
python -m training.ppo.scripts.score_hps \
    --images samples/test_rl \
    --pattern "**/*.png" \
    --prompts src/prompts/test100.txt \
    --weights weights/HPS_v2.1_compressed.pt