RLEPD — PPO Fine-Tuning for EPD Predictor Tables
================================================

This repository implements PPO fine-tuning for EPD predictor tables on top of Stable Diffusion v1.5. The end-to-end workflow has two entry points:

1. `train.sh` — create an initial predictor table, run `training/ppo.launch`, and export the learned predictor.
2. `sample.sh` — run baseline solvers (DDIM/DPM2/EDM/iPNDM), sample with the PPO-trained predictor, and score the generated images (HPS/CLIP/Aesthetic/PickScore/ImageReward/MPS).

Directory Overview
------------------

| Path | Purpose |
|------|---------|
| `training/ppo/` | PPO configs, policy/runner/trainer, reward adapters, export utilities, and evaluation scripts. |
| `training/network*`, `solvers.py`, `solver_utils.py` | EPD predictor and solver implementations shared by training and sampling. |
| `sample.py`, `sample_baseline.py` | CLI entrypoints for PPO sampling and baseline solvers. |
| `train.sh`, `sample.sh` | Reference pipelines for training and sampling/evaluation. |
| `HPSv2`, `MPS` | Minimal code required by the reward and MPS scoring scripts. |
| `torch_utils/`, `dnnlib/` | Utilities for distributed launch, weight download, and serialization. |

Environment
-----------

- Python ≥ 3.9 and PyTorch ≥ 1.13 with CUDA.
- Install dependencies listed in `environment.yml` (PyTorch, click, omegaconf, huggingface_hub, torchvision, etc.) or construct an equivalent virtual environment manually.
- `torch_utils.download_util.check_file_by_key` fetches the Stable Diffusion v1.5 checkpoint and MS-COCO prompts automatically.
- HPS weights are downloaded from `xswu/HPSv2` on first use; set `HPS_ROOT` to override the cache path.

Key Components
--------------

- `training/ppo/launch.py`  
  - Reads YAML configs (e.g., `training/ppo/cfgs/sd15_parallel.yaml`), builds `EPDParamPolicy`, `EPDRolloutRunner`, `RewardHPS`, and `PPOTrainer`.  
  - Writes logs to `<run_dir>/logs/metrics.jsonl` and checkpoints to `<run_dir>/checkpoints/`.

- `training/ppo/rl_runner.EPDRolloutRunner`  
  - Samples Dirichlet tables from the policy, wraps them as an EPD predictor, and invokes the Stable Diffusion model to generate images.

- `training/ppo/policy.EPDParamPolicy`  
  - Produces Dirichlet parameters for positions/weights per diffusion step and exposes `mean_table()` / `sample_table()` to convert them into predictor tables.

- `training/ppo/export_epd_predictor`  
  - CLI: `python -m training.ppo.export_epd_predictor <run_dir> --checkpoint checkpoints/policy-stepXXXXXX.pt`.  
  - Outputs `export/network-snapshot-export-step*.pkl` (used by `sample.py`) and `export/training_options.json`.

- `sample.py` / `sample_baseline.py`  
  - `sample.py` loads an exported predictor and runs PPO-tuned sampling.  
  - `sample_baseline.py` runs handcrafted solvers (DDIM/DPM2/EDM/iPNDM).  
  - Both rely on `torch_utils.distributed` for multi-GPU launches via `torchrun`.

Training Workflow
-----------------

```bash
# Optional: create an initial predictor table
python fake_train.py \
  --num-steps 11 \
  --num-points 2 \
  --outdir exps/99999-ms_coco-11-20-epd-dpm-1-discrete \
  --snapshot-step 99999

# Run PPO (adjust nproc_per_node/ports as needed)
torchrun --nproc_per_node=8 --master_port=59500 -m training.ppo.launch \
  --config training/ppo/cfgs/sd15_parallel.yaml

# Export predictor weights for sampling
python -m training.ppo.export_epd_predictor \
  exps/<run-id> \
  --checkpoint checkpoints/policy-step008000.pt
```

Sampling Workflow
-----------------

```bash
# Baseline solvers
python sample_baseline.py --sampler ddim \
  --dataset-name ms_coco \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" --batch 16 \
  --num-steps 37 --schedule-type time_uniform --schedule-rho 1.0 \
  --outdir ./samples/test_ddim_nfe36

# PPO / EPD sampling
MASTER_PORT=29600 python sample.py \
  --predictor_path exps/<run-id>/export/network-snapshot-export-step008000.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-999" \
  --batch 16 \
  --outdir ./samples/<your-exp>
```

Evaluation Workflow
-------------------

Scoring scripts live under `training/ppo/scripts/` and share a uniform interface:

```bash
python -m training.ppo.scripts.score_hps \
  --images path/to/images \
  --pattern "**/*.png" \
  --prompts src/prompts/test.txt \
  --weights weights/HPS_v2.1_compressed.pt \
  --output-json results/<exp>-hps.json
```

Available metrics:

| Script | Description |
|--------|-------------|
| `score_hps.py` | Human Preference Score v2 (same reward used during PPO). |
| `score_clip.py` | CLIPScore. |
| `score_aesthetic.py` | Aesthetic predictor (sac+logos+ava1). |
| `score_pick.py` | PickScore. |
| `score_imagereward.py` | ImageReward. |
| `score_mps.py` | Multi-dimensional Preference Score (requires `MPS/trainer/models`). |

Each script prints summary statistics and writes them to JSON for downstream comparisons.

Customization Notes
-------------------

- **Rewards**: see `training/ppo/reward_multi.py` to mix multiple metrics, or implement a new adapter following `RewardHPS`.  
- **Prompts / Datasets**: current configs target MS-COCO / Stable Diffusion v1.5. To support other models, extend `sample.py` and `sample_baseline.py` with additional `create_model` branches.  
- **Sampling scripts**: extend `sample.sh` with additional scorers or copy the commands into your own automation scripts; each scorer accepts `--output-json` for logging.

Running `train.sh` and `sample.sh` is sufficient to reproduce the full training, sampling, and evaluation pipeline described above.
