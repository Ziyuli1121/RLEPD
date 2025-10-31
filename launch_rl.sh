#!/usr/bin/env bash

# Stage 9+ 常用命令备忘脚本（本脚本不再自动执行命令，只列示参考）

# 1. 训练（可添加额外 override）
python -m training.ppo.launch \
    --config training/ppo/cfgs/sd15_base.yaml

# 2. 导出策略均值为 EPD predictor
python -m training.ppo.export_epd_predictor \
    exps/<run-id> \
    --checkpoint checkpoints/policy-step000500.pt

# 3. 使用导出的 predictor 生成图像
python sample.py \
    --predictor_path exps/<run-id>/export/network-snapshot-export-step000500.pkl \
    --seeds 0-3 --batch 4 --prompt "a photo of a small corgi"

# 4. 评估生成图像的 HPS 分数
python -m training.ppo.scripts.score_hps \
    --images exps/<run-id>/samples \
    --prompts path/to/prompts.csv \
    --weights weights/HPS_v2.1_compressed.pt
