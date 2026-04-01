## Parallel Diffusion Solver via Residual Dirichlet Policy Optimization <br><sub>Official implementation</sub>

## Requirements

This repository uses [EDM](https://github.com/NVlabs/edm) as the base training structure, but the active RLEPD pipeline in this checkout is:

`fake_train -> RDPO train -> export_epd_predictor -> sample -> multi-metric evaluation`

The current code expects:

- Python `3.10` for the generic repo path that relies on the vendored `diffusers/`
- PyTorch `2.x`
- single-node GPU execution for training / sampling
- local evaluation weights under `./weights`

For FLUX.1-dev formal single-node runs, this checkout also maintains a validated Python `3.9` path via [environment.flux.yml](./environment.flux.yml), using an installed `diffusers` package that already exposes `FluxPipeline`.

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
4. Sample with `sample.py`, `sample_sd3.py`, `sample_flux.py`, or `sample_flux_baseline.py`
5. Evaluate with the score CLIs or the helper shell scripts

Useful entry scripts:

- [train.sh](./train.sh): cold-start + RL train + export examples
- [train_flux.sh](./train_flux.sh): dedicated FLUX.1-dev cold-start + RL train + export entrypoint
- [launch.sh](./launch.sh): end-to-end launch examples
- [sample_flux.sh](./sample_flux.sh): FLUX.1-dev EPD replay example
- [sample_flux_baseline.sh](./sample_flux_baseline.sh), [test_flux.sh](./test_flux.sh): FLUX.1-dev baseline/formal-eval examples
- [compare_flux_euler.py](./compare_flux_euler.py): internal debug helper for comparing official FLUX Euler vs RLEPD `ddim(flowmatch)`
- [sample.sh](./sample.sh), [sample512.sh](./sample512.sh), [sample1024.sh](./sample1024.sh): sampling + scoring examples
- [test_regression.sh](./test_regression.sh): official regression-style `export -> sample -> score`
- [test_15.sh](./test_15.sh), [test_512.sh](./test_512.sh), [test_1024.sh](./test_1024.sh): experiment-specific loops
- [TEST.sh](./TEST.sh): GPU smoke test for `sd1.5`, `sd3-512`, `sd3-1024`, and `flux`
- [environment.flux.yml](./environment.flux.yml): pinned FLUX single-node runtime based on the validated `epd` environment

## FLUX.1-dev Notes

- Supported FLUX variant in this checkout: `black-forest-labs/FLUX.1-dev`
- Resolution is fixed to `1024x1024`
- EPD replay entrypoint is [sample_flux.py](./sample_flux.py)
- Baseline sampler entrypoint is [sample_flux_baseline.py](./sample_flux_baseline.py)
- Official FLUX Euler baseline means the native diffusers `FluxPipeline + FlowMatchEulerDiscreteScheduler`
- In this repo, the closest project-side Euler-style comparator is `ddim` with `schedule_type=flowmatch`
- Training config is [training/ppo/cfgs/flux_dev.yaml](./training/ppo/cfgs/flux_dev.yaml)
- FLUX reuses the same RDPO/PPO/export pipeline as SD1.5 and SD3
- Current FLUX guidance in this repo uses embedded `guidance_scale=3.5`; true CFG / negative prompt branches are not wired into the RLEPD FLUX path
- FLUX artifacts are expected to carry concrete `sigma_min / sigma_max / flowmatch_mu / flowmatch_shift` metadata; the formal FLUX path no longer relies on runtime-only schedule derivation for new artifacts
- On Python `3.9`, the FLUX path requires an installed `diffusers` that already exposes `FluxPipeline`
- If your Python `3.9` environment does not provide `FluxPipeline`, run FLUX with Python `3.10+` and the repo's newer vendored `diffusers`
- The official FLUX baseline path should treat `output_type="pt"` as an already postprocessed `[0, 1]` tensor; do not re-normalize it
- Do not wrap the official FLUX pipeline call in the default CUDA autocast context; this checkout loads FLUX in BF16 and mismatched autocast can produce invalid outputs
- `test_flux.sh` is the formal FLUX evaluation wrapper:
  - it exports the latest predictor
  - samples FLUX EPD images
  - aligns prompt count to the requested seeds
  - runs full `clip / hps / aesthetic / imagereward / mps` on the EPD sample
  - runs `PickScore` only when local weights exist or `ENABLE_PICKSCORE_DOWNLOAD=1`
  - baseline solver sweep remains generation-only by default
- FLUX runtime preflight is intentionally lightweight: it checks the known-good package family, local/remote model reachability, and that the installed `diffusers` package contains the FLUX pipeline files needed by this checkout
- Treat official FLUX Euler and project-side `ddim(flowmatch)` as different baselines; they are close, but not strictly equivalent

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
- FLUX.1-dev (1024*1024)

The exported and pre-generated predictors in this checkout live under `./exps/`.
