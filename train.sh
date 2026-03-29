# 伪训练（设置初始值）
python fake_train.py \
  --num-steps 11 \
  --num-points 2 \
  --outdir exps/f15 \
  --r-base 0.5 \
  --r-epsilon 0.33 \
  --scale-dir 0.05 \
  --scale-time 0.05


# 开始RL训练
torchrun --master_port=12312 --nproc_per_node=1 -m training.ppo.launch \
    --config training/ppo/cfgs/sd15_k5.yaml

torchrun --master_port=23123 --nproc_per_node=1 -m training.ppo.launch \
    --config training/ppo/cfgs/sd15_k20.yaml

torchrun --master_port=31231 --nproc_per_node=1 -m training.ppo.launch \
    --config training/ppo/cfgs/sd15_k50.yaml


# 导出策略均值为 EPD predictor
python -m training.ppo.export_epd_predictor \
    exps/20251123-215008-sd3_smoke \
    --checkpoint checkpoints/policy-step002900.pt













# sd3
python fake_train.py \
  --outdir exps/f512 \
  --num-steps 11 \
  --num-points 2 \
  --guidance-rate 4.5 \
  --schedule-type flowmatch \
  --backend sd3 \
  --resolution 512 \
  --backend-options '{"model_name_or_path":"stabilityai/stable-diffusion-3-medium-diffusers"}' \
  --r-base 0.5 \
  --r-epsilon 0.33
  

python fake_train.py \
  --outdir exps/f1024 \
  --num-steps 11 \
  --num-points 2 \
  --guidance-rate 4.5 \
  --schedule-type flowmatch \
  --backend sd3 \
  --resolution 1024 \
  --backend-options '{"model_name_or_path":"stabilityai/stable-diffusion-3-medium-diffusers"}' \
  --r-base 0.5 \
  --r-epsilon 0.33


torchrun --master_port=33333 --nproc_per_node=1 -m training.ppo.launch \
    --config training/ppo/cfgs/sd3_512.yaml

torchrun --master_port=44444 --nproc_per_node=1 -m training.ppo.launch \
    --config training/ppo/cfgs/sd3_1024.yaml

# /work/nvme/betk/zli42/RLEPD/exps/20251204-144911-sd3_1024/logs/metrics.jsonl
# /work/nvme/betk/zli42/RLEPD/exps/20251203-011623-sd3_512/logs/metrics.jsonl
python exp_visuals/scripts/plot_metrics.py \
    --metrics-file /work/nvme/betk/zli42/RLEPD/exps/20251219-032038-sd3_512_new/logs/metrics.jsonl \
    --metrics hps_mean \
    --smooth-window 1000 \
    --output exp_visuals/512/hps_mean.png

python exp_visuals/scripts/plot_metrics.py \
    --metrics-file /work/nvme/betk/zli42/RLEPD/exps/20251219-032043-sd3_1024_continue/logs/metrics.jsonl \
    --metrics hps_mean \
    --smooth-window 1000 \
    --output exp_visuals/1024/hps_mean.png


python -m training.ppo.export_epd_predictor \
  exps/20251210-005434-sd3_512 \
  --checkpoint checkpoints/policy-step008000.pt
