# 伪训练（设置初始值）
python fake_train.py \
  --num-steps 11 \
  --num-points 2 \
  --outdir exps/fake-sd15-11 \
  --r-base 0.5 \
  --r-epsilon 0.33


# 开始RL训练
torchrun --master_port=29500 --nproc_per_node=1 -m training.ppo.launch \
    --config training/ppo/cfgs/sd15.yaml

# 导出策略均值为 EPD predictor
python -m training.ppo.export_epd_predictor \
    exps/20251123-215008-sd3_smoke \
    --checkpoint checkpoints/policy-step002900.pt













# sd3
python fake_train.py \
  --outdir exps/fake-sd3-11-512 \
  --num-steps 12 \
  --num-points 2 \
  --guidance-rate 4.5 \
  --schedule-type flowmatch \
  --backend sd3 \
  --resolution 512 \
  --backend-options '{"model_name_or_path":"stabilityai/stable-diffusion-3-medium-diffusers"}' \
  --r-base 0.5 \
  --r-epsilon 0.33
  

python fake_train.py \
  --outdir exps/fake-sd3-11-1024-new \
  --num-steps 12 \
  --num-points 2 \
  --guidance-rate 4.5 \
  --schedule-type flowmatch \
  --backend sd3 \
  --resolution 1024 \
  --backend-options '{"model_name_or_path":"stabilityai/stable-diffusion-3-medium-diffusers"}' \
  --r-base 0.5 \
  --r-epsilon 0.33


torchrun --master_port=22222 --nproc_per_node=1 -m training.ppo.launch \
    --config training/ppo/cfgs/sd3_512.yaml

torchrun --master_port=12345 --nproc_per_node=1 -m training.ppo.launch \
    --config training/ppo/cfgs/sd3_1024.yaml

# /work/nvme/betk/zli42/RLEPD/exps/20251204-144911-sd3_1024/logs/metrics.jsonl
# /work/nvme/betk/zli42/RLEPD/exps/20251203-011623-sd3_512/logs/metrics.jsonl
python exp_visuals/scripts/plot_metrics.py \
    --metrics-file /work/nvme/betk/zli42/RLEPD/exps/20251210-005434-sd3_512/logs/metrics.jsonl \
    --metrics hps_mean \
    --smooth-window 1000 \
    --output exp_visuals/sd3_512_new/hps_mean.png

python exp_visuals/scripts/plot_metrics.py \
    --metrics-file /work/nvme/betk/zli42/RLEPD/exps/20251210-005434-sd3_512/logs/metrics.jsonl \
    --metrics kl \
    --smooth-window 0 \
    --output exp_visuals/sd3_512_new/kl.png

python exp_visuals/scripts/plot_metrics.py \
    --metrics-file /work/nvme/betk/zli42/RLEPD/exps/20251210-011145-sd3_1024/logs/metrics.jsonl \
    --metrics hps_mean \
    --smooth-window 1000 \
    --output exp_visuals/sd3_1024_new/hps_mean.png

python exp_visuals/scripts/plot_metrics.py \
    --metrics-file /work/nvme/betk/zli42/RLEPD/exps/20251210-011145-sd3_1024/logs/metrics.jsonl \
    --metrics kl \
    --smooth-window 0 \
    --output exp_visuals/sd3_1024_new/kl.png

python exp_visuals/scripts/plot_metrics.py \
    --metrics-file /work/nvme/betk/zli42/RLEPD/exps/20251212-225028-sd15/logs/metrics.jsonl \
    --metrics hps_mean \
    --smooth-window 200 \
    --output exp_visuals/sd15/hps_mean.png

python exp_visuals/scripts/plot_metrics.py \
    --metrics-file /work/nvme/betk/zli42/RLEPD/exps/20251212-225028-sd15/logs/metrics.jsonl \
    --metrics kl \
    --smooth-window 0 \
    --output exp_visuals/sd15/kl.png


python -m training.ppo.export_epd_predictor \
  exps/20251210-005434-sd3_512 \
  --checkpoint checkpoints/policy-step008000.pt
