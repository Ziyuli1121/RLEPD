## Parallel Diffusion Sampling with Low-Dimensional Alignment <br><sub>Official implementation</sub>

**Abstract**: xxxxxxxxxxxxxxxxxxxx

## Requirements

This codebase mainly refers to the codebase of [EDM](https://github.com/NVlabs/edm) as the base environment and then add other required packages. 

```bash
conda env create -f environment.yml -n edm
conda activate edm
pip install omegaconf
pip install gdown
conda install lightning -c conda-forge
pip install git+https://github.com/openai/CLIP.git
pip install transformers
pip install taming-transformers
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install kornia fairscale piq
pip install accelerator
pip install --upgrade diffusers[torch]
cd src
mkdir ms_coco
cd ms_coco
wget https://huggingface.co/dnwalkup/StableDiffusion-v1-Releases/resolve/main/v1-5-pruned-emaonly.ckpt
python download_hpsv2_weights.py --version v2.1 #https://huggingface.co/xswu/HPSv2/blob/main/HPS_v2.1_compressed.pt
cd ..
mkdir weights
cd weights
# Download evaluators
wget https://huggingface.co/haor/aesthetics/resolve/main/sac%2Blogos%2Bava1-l14-linearMSE.pth
gdown 17qrK_aJkVNM75ZEvMEePpLj6L867MLkN
```

## Implementation Guide

- Run the commands in [launch.sh](./launch.sh) for RL pipeline and sampling.
- Complete parameter descriptions are available in the next section.

### Example Commands:

```bash
# RL training
xxxxx
```

```bash
# Sampling
xxxxx
```

## Parameter Description

| Category          | Parameter          | Default | Description |
|-------------------|--------------------|---------|-------------|
| **General Options** | `dataset_name`     | None    | Supported datasets: `['cifar10', 'ffhq', 'afhqv2', 'imagenet64', 'lsun_bedroom', 'imagenet256', 'lsun_bedroom_ldm', 'ms_coco']` |
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
|                   | `afs`              | False   | Enable Accelerated First Step (saves initial model evaluation) |
| **Schedule Flags**| `schedule_type`    | 'polynomial' | Time discretization: `['polynomial', 'logsnr', 'time_uniform', 'discrete']` |
|                   | `schedule_rho`     | 7       | Time step exponent (required for `polynomial`, `time_uniform`, `discrete`) |
| **Additional Flags** | `max_order`       | None    | Multi-step solver order: `1-4` for iPNDM, `1-3` for DPM-Solver++ |
|                   | `predict_x0`       | True    | DPM-Solver++: Use data prediction formulation |
|                   | `lower_order_final`| True    | DPM-Solver++: Reduce order at final sampling stages |
| **Guidance Flags** | `guidance_type`    | None    | Guidance method: `['cg' (classifier), 'cfg' (classifier-free), 'uncond' (unconditional), None]` |
|                   | `guidance_rate`    | None    | Guidance strength parameter |
|                   | `prompt`           | None    | Text prompt for Stable Diffusion sampling |

### EPD Step Calculation
When `num_steps=N`, total steps = `2*(N-1)` (EPD inserts intermediate steps)

## Pre-trained EPD Predictors

We provide pre-trained EPD predictors for:

- SD1.5
- SD3-Medium (512*512)
- SD3-Medium (1024*1024)

The pre-trained EPD predictors are available in `./exp/`.

## Citation
If you find this repository useful, please consider citing the following paper:

```

```