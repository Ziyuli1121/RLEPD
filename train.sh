# 伪训练（设置初始值）
python fake_train.py \
  --num-steps 11 \
  --num-points 2 \
  --outdir exps/00000-ms_coco-11-20-epd-dpm-1-discrete \
  --snapshot-step 99999 \
  --run-dir ./exps/00000-ms_coco-11-20-epd-dpm-1-discrete


# 开始RL训练
torchrun --master_port=59500 --nproc_per_node=1 -m training.ppo.launch \
    --config training/ppo/cfgs/sd15_base.yaml

# 导出策略均值为 EPD predictor
python -m training.ppo.export_epd_predictor \
    exps/20251115-110803-sd15_rl_base \
    --checkpoint checkpoints/policy-step008000.pt

