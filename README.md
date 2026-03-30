## Parallel Diffusion Solver via Residual Dirichlet Policy Optimization <br><sub>Official implementation</sub>

## Requirements

This repository uses [EDM](https://github.com/NVlabs/edm) as the base training structure, but the active RLEPD pipeline in this checkout is:

`fake_train -> RDPO train -> export_epd_predictor -> sample -> multi-metric evaluation`

The current code expects:

- Python `3.10`
- PyTorch `2.x`
- single-node GPU execution for training / sampling
- local evaluation weights under `./weights`

The bundled `diffusers/` and `HPSv2/` directories are auto-discovered by the updated code paths. You do not need to set `PYTHONPATH` manually for them.

```bash
conda env create -f environment.yml -n edm
conda activate edm
```

Download the runtime assets used by the main pipeline:

```bash
mkdir -p src/ms_coco
mkdir -p weights

# Stable Diffusion 1.5 backbone
wget -O src/ms_coco/v1-5-pruned-emaonly.ckpt \
  https://huggingface.co/dnwalkup/StableDiffusion-v1-Releases/resolve/main/v1-5-pruned-emaonly.ckpt

# Evaluation weights
wget -O weights/HPS_v2.1_compressed.pt \
  https://huggingface.co/xswu/HPSv2/resolve/main/HPS_v2.1_compressed.pt
wget -O weights/sac+logos+ava1-l14-linearMSE.pth \
  https://huggingface.co/haor/aesthetics/resolve/main/sac%2Blogos%2Bava1-l14-linearMSE.pth
# Place the ImageReward checkpoint at weights/ImageReward.pt
```

Optional:

- If you have a local PickScore checkpoint, place it at `weights/PickScore_v1/`
- If you use local SD3 model snapshots, point the SD3 configs or `--backend-config/--model-id` arguments at them

## Pipeline Guide

1. Generate a cold-start EPD table with `fake_train.py`
2. Run RDPO training with `python -m training.ppo.launch`
3. Export the trained policy mean with `python -m training.ppo.export_epd_predictor`
4. Sample with `sample.py` or `sample_sd3.py`
5. Evaluate with the score CLIs or the helper shell scripts

Useful entry scripts:

- [train.sh](./train.sh): cold-start + RL train + export examples
- [launch.sh](./launch.sh): end-to-end launch examples
- [sample.sh](./sample.sh), [sample512.sh](./sample512.sh), [sample1024.sh](./sample1024.sh): sampling + scoring examples
- [test_regression.sh](./test_regression.sh): official regression-style `export -> sample -> score`
- [test_15.sh](./test_15.sh), [test_512.sh](./test_512.sh), [test_1024.sh](./test_1024.sh): experiment-specific loops

## Parameter Description

| Category          | Parameter          | Default | Description |
|-------------------|--------------------|---------|-------------|
| **General Options** | `dataset_name`     | None    | This trimmed RLEPD checkout officially supports `ms_coco` |
|                   | `predictor_path`   | None    | Path or experiment number of trained EPD predictor |
|                   | `batch`            | 64      | Total batch size |
|                   | `seeds`            | "0-63"  | Random seed range for image generation |
|                   | `grid`             | False   | Organize output images in grid layout |
|                   | `total_kimg`       | 10      | Training duration (in thousands of images) |
|                   | `scale_dir`        | 0.05    | Gradient direction scale (`c_n` in paper). Range: `[1-scale_dir, 1+scale_dir]` |
|                   | `scale_time`       | 0.05       | Input time scale (`a_n` in paper). Range: `[1-scale_time, 1+scale_time]` |
| **Solver Flags**  | `sampler_stu`      | 'epd'   | Student solver: `['epd', 'ipndm']` |
|                   | `sampler_tea`      | 'dpm'   | Teacher solver type |
|                   | `num_steps`        | 4       | Initial timestamps for student solver. Final steps = `2*(num_steps-1)` (EPD inserts intermediate steps) |
|                   | `M`                | 3       | Intermediate steps inserted between teacher solver steps |

### EPD Step Calculation
When `num_steps=N`, total steps = `2*(N-1)` (EPD inserts intermediate steps)

## Pre-trained EPD Predictors

We provide pre-trained EPD predictors for:

- SD1.5
- SD3-Medium (512*512)
- SD3-Medium (1024*1024)

The exported and pre-generated predictors in this checkout live under `./exps/`.
