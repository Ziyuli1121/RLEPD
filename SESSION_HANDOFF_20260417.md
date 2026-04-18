# RLEPD Session Handoff

Last updated: 2026-04-17

This document is a session-specific migration report for future Codex sessions. It summarizes the current repo state and the decisions made across this conversation, beyond the durable background already captured in `CONTEXT.md` and `README.md`.

## 1. Executive Summary

RLEPD is currently being used as a solver-learning pipeline for text-to-image sampling across:

- `sd1.5`
- `sd3-512`
- `sd3-1024`
- `FLUX.1-dev`

The canonical pipeline remains:

1. `fake_train.py`
2. `python -m training.ppo.launch`
3. `python -m training.ppo.export_epd_predictor`
4. `sample.py` / `sample_sd3.py` / `sample_flux.py`
5. `training/ppo/scripts/score_*.py`

The most important session-specific facts are:

- FLUX formal RL training is intentionally using the historical intermediate algorithm state between commits `3c58295` and `29987d4`, not standard same-prompt RLOO.
- Formal COCO-10k FID now exists in-repo via a clean-fid based scorer.
- A second legacy auxiliary FID path also exists in-repo via `src/ms_coco-512x512.npz`; this is a different protocol and must never be mixed with the formal clean-fid result.
- `taming-transformers` support for SD1.5 is currently fragile because the repo prefers a local `src/taming-transformers` source tree, and the active repo copy is empty.

## 2. Current Intended Training Semantics

### 2.1 High-level model

The project RL-trains an EPD solver table, not a full diffusion backbone. The policy lives around a cold-start predictor table:

- `r`: intermediate positions inside each coarse step
- `weight`: simplex weights over candidate directions
- optional `scale_dir`, `scale_time`

Training learns residual corrections on top of that table and later exports the policy mean back into predictor format.

Important files:

- `fake_train.py`
- `training/ppo/launch.py`
- `training/ppo/export_epd_predictor.py`
- `training/networks.py`
- `solvers.py`

### 2.2 FLUX formal RL semantics

The intended FLUX algorithm is the historical intermediate state between:

- `3c58295b7ab7b564ebc3a3337680704b8eec0a70`
- `29987d46524d65637f8b9f1daddf75922943fb61`

Meaning:

- rollout prompt layout is `prompt-major`
- PPO advantage is still the legacy mixed-prompt leave-one-out baseline
- this is not standard same-prompt RLOO

Concrete consequence:

- with `rollout_batch_size=8` and `rloo_k=4`
- generation layout is `2 prompts x 4 images`
- but advantage groups are still mixed in the old historical way

Current FLUX training source of truth:

- `training/ppo/cfgs/flux_dev.yaml`
- `training/ppo/ppo_trainer.py`
- `training/ppo/rl_runner.py`
- `train_flux.sh`

### 2.3 Current FLUX training defaults

From `train_flux.sh` and `training/ppo/cfgs/flux_dev.yaml`:

- single-card launch uses `python -m training.ppo.launch`
- multi-card launch only when `FLUX_NPROC_PER_NODE > 1`
- `reward.batch_size = 1`
- `rollout_batch_size = 8`
- `rloo_k = 4`
- `minibatch_size = 4`
- `ppo_epochs = 1`
- `learning_rate = 7e-5`
- fake-train defaults: `num_steps = 9`, `num_points = 2`, `guidance_rate = 3.5`

Important runtime flags in `train_flux.sh`:

- `FLUX_CUDA_LAUNCH_BLOCKING` defaults to `1`
- `FLUX_TORCH_SHOW_CPP_STACKTRACES` defaults to `1`

This was introduced for debugging a prior FLUX segfault around step ~1090. It improves diagnosis but can slow training.

## 3. Sampling Conventions and Solver Semantics

### 3.1 Main sampler entrypoints

- `sample.py`: SD1.5 EPD replay
- `sample_baseline.py`: SD1.5 baseline samplers
- `sample_sd3.py`: SD3 EPD replay
- `sample_sd3_baseline.py`: SD3 baseline samplers
- `sample_flux.py`: FLUX EPD replay
- `sample_flux_baseline.py`: FLUX baseline samplers

### 3.2 Prompt file conventions

Two important prompt artifacts now exist:

- `src/prompts/coco10k.csv`
- `src/prompts/coco10k.txt`

These were verified to be strictly aligned:

- `coco10k.csv` has 10000 rows
- `coco10k.txt` has 10000 prompts
- `txt[i]` matches `csv.caption[i]`

Use:

- `coco10k.csv` as the formal protocol source of truth
- `coco10k.txt` as the convenient prompt-only input for generation

Important prompt/seed mapping caveat:

- SD1.5 samplers (`sample.py`, `sample_baseline.py`) effectively map prompts by seed-value slice
- SD3 / FLUX samplers first read `len(seeds)` prompts and then consume them in seed-list order
- this means contiguous seed ranges like `0-9999` behave consistently across backbones, but sparse/non-contiguous seed lists can change prompt alignment semantics between SD1.5 and SD3/FLUX

### 3.3 Baseline solver step/NFE conventions

For baseline scripts, the step/NFE conventions in this repo are:

- Euler-like single-step methods: one new model evaluation per step/interval family
- `ipndm`: approximately `steps - 1` new model evaluations
- `dpm2`: approximately `2 * (steps - 1)`
- `edm`: approximately `2 * steps - 1`

For FLUX latency work, the benchmark script reports both nominal NFE and estimated forward-equivalent counts:

- `benchmark_flux_solver_latency.py`

Important interpretation:

- `epd_parallel` is the correct wall-clock latency path for EPD
- EPD nominal NFE and actual compute-equivalent forward count are not the same
- e.g. a `9-step`, `num_points=2` EPD predictor is nominally “16 NFE” but roughly “24 point-evals”

## 4. Formal COCO-10k Evaluation Protocols

There are now two separate FID protocols in the repo. They must remain separate in analysis and reporting.

### 4.1 Formal clean FID

Scorer:

- `training/ppo/scripts/score_fid_dir.py`

Helper:

- `scripts/pipeline_common.sh`
- function: `score_fid_dir`

Protocol:

- fake images from a generated sample directory
- manifest from `src/prompts/coco10k.csv`
- real images from `src/coco10k_real_val2014`
- clean-fid
- `mode=clean`
- `eval_res=256`
- generation itself remains native resolution

Formal name:

- `FID-10k (fixed COCO subset, clean preprocessing, native 1024 generation)`

Real subset preparation artifacts already exist:

- `src/val2014`
- `src/coco10k_real_val2014`
- `training/ppo/scripts/prepare_coco_real_subset.py`

Current formal clean FID result already obtained for FLUX EPD:

- `results/flux_fid_20260412/flux_epd_step002800_coco10k_fid.json`
- value was approximately `26.651150395357035`

### 4.2 Legacy auxiliary NPZ FID

Scorer:

- `training/ppo/scripts/score_fid_npz_dir.py`

Helper:

- `scripts/pipeline_common.sh`
- function: `score_fid_npz_dir`

Reference stats:

- `src/ms_coco-512x512.npz`

Protocol:

- fake image directory only
- recursively collect `**/*.png`
- strict default requirement: exactly `10000` PNG files
- NVIDIA/EDM-style Inception pkl
- standard Fréchet distance against precomputed `mu/sigma`

Formal name inside the script:

- `legacy_fid_ms_coco_512x512_npz`

This is not the same thing as the clean-fid COCO-10k metric and should not share a single “FID” column.

## 5. Reward / Preference Metric Evaluation

Existing score scripts:

- `training/ppo/scripts/score_clip.py`
- `training/ppo/scripts/score_hps.py`
- `training/ppo/scripts/score_aesthetic.py`
- `training/ppo/scripts/score_pick.py`
- `training/ppo/scripts/score_imagereward.py`
- `training/ppo/scripts/score_mps.py`

Main helper:

- `scripts/pipeline_common.sh`
- function: `score_all_metrics_dir`

Notes:

- local weights are preferred
- PickScore falls back to HF download when local weights are absent
- prompt/image alignment matters; some wrappers generate prompt subsets for this reason

## 6. Latency Benchmarking

Main script:

- `benchmark_flux_solver_latency.py`

Current intended usage:

- compare FLUX baseline solvers and optionally EPD on the same prompt/seed subset
- warm up each solver before timing
- report both:
  - sampling-only latency
  - end-to-end latency

The script now supports baseline-only benchmarking:

- if no `--epd-predictor` and no `--epd-run-dir` are supplied, it only benchmarks baselines

Important output artifacts:

- `summary.json`
- `latency_records.csv`

Already existing result dirs include:

- `results/flux_solver_latency_step001000_16vs16`
- `results/flux_solver_latency_step001000_16vs20`
- other similar latency result dirs under `results/`

## 7. Current Best-Model / Wrapper Scripts

Root wrappers currently present:

- `sd15_fid.sh`
- `sd3-512_fid.sh`
- `sd3-1024_fid.sh`

Current state:

- they run best-model predictor sampling for COCO-10k
- they also now append baseline solver generation commands
- GPU device is hardcoded inline using `CUDA_VISIBLE_DEVICES=... python ...`
- they do not parse arguments anymore
- they are convenience experiment wrappers, not a clean canonical CLI surface

Important caveat:

- `sd15_fid.sh` currently contains:
  - best-model SD1.5 generation
  - one full-run `ddim` 10k command at `50` steps
  - two shard-split `ddim` commands sharing the same outdir with distinct `MASTER_PORT`s
  - `edm` at `25` steps
  - `dpm2` at `25` steps
  - `ipndm` at `50` steps
- `sd3-512_fid.sh` currently contains:
  - best-model SD3-512 generation
  - `sd3`/Euler at `28` steps
  - `edm` at `14` steps
  - `dpm2` at `14` steps
  - `ipndm` at `28` steps
- `sd3-1024_fid.sh` is currently only partially active:
  - best-model SD3-1024 generation is commented out
  - `sd3`/Euler is commented out
  - `edm` is commented out
  - only `dpm2` at `14` steps and `ipndm` at `28` steps are active
- so `sd3-1024_fid.sh` should be treated as a partially edited experiment wrapper, not a finished symmetric counterpart to the other two scripts

Current git-dirty files at the time of this report:

- `fid_legacy.sh`
- `sd15_fid.sh`
- `sd3-1024_fid.sh`
- `sd3-512_fid.sh`
- untracked: `SESSION_HANDOFF_20260417.md`
- untracked: `src/prompts/coco10k_9400_9999.txt`

## 8. Environment and Runtime Pitfalls

### 8.1 `taming-transformers`

This is currently the most important SD1.5 environment hazard.

Facts:

- the repo prefers a local `src/taming-transformers` source tree via `bootstrap_local_taming()`
- SD1.5 / LDM code paths expect `import taming`
- current `src/taming-transformers` in this repo is empty
- the active `epd` environment previously showed a broken editable install residue pointing to that empty directory

Implication:

- `import taming` may work inside a clone/source directory due to current working directory on `sys.path`
- that does not mean `taming` is correctly installed in the whole conda environment
- leaving that directory can make imports fail again

Bottom line:

- for stable SD1.5 work, either restore a real `src/taming-transformers` source tree or stop prioritizing the local path
- a random `pip install taming-transformers` is not enough unless the repo’s local bootstrap behavior is also accounted for

### 8.2 Submodule / gitignore behavior

Earlier investigation showed `src/taming-transformers` behaved as a tracked gitlink/submodule path in the parent repo. `.gitignore` does not hide dirty state for tracked submodule entries.

### 8.3 `CUDA_VISIBLE_DEVICES` and `MASTER_PORT`

For sampling scripts like `sample_baseline.py`:

- `CUDA_VISIBLE_DEVICES` only selects what the process can see
- it does not create automatic multi-GPU parallelism

For manual multi-process sharding:

- split seeds explicitly
- if the script calls `torch_utils.distributed.init()`, use different `MASTER_PORT`s per independent process

Example pattern:

- process A: `CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29501 ...`
- process B: `CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29502 ...`

### 8.4 Sample directory contamination

This was a real issue in this session.

Problem:

- scoring paths recurse on `**/*.png`
- Jupyter `.ipynb_checkpoints` can create hidden PNG copies
- those inflate counts, e.g. a supposed 10000-image directory can become 10002

Observed example:

- `samples/sd3_512_euler_coco10k/000000/.ipynb_checkpoints/000010-checkpoint.png`
- `samples/sd3_512_euler_coco10k/002000/.ipynb_checkpoints/002009-checkpoint.png`

Recommended cleanup before any recursive metric:

```bash
find samples -type d -name '.ipynb_checkpoints' -exec rm -rf {} +
```

Then recheck counts:

```bash
find samples/<experiment> -type f -name '*.png' | wc -l
```

### 8.5 `pipeline_common.sh` sourcing guard

`scripts/pipeline_common.sh` uses a guard variable:

- `RLEPD_PIPELINE_COMMON_SH`

If you edit the file and re-source it in the same shell, the new functions will not appear unless you first run:

```bash
unset RLEPD_PIPELINE_COMMON_SH
source scripts/pipeline_common.sh
```

## 9. Important Existing Artifacts and Results

Key FLUX run dirs:

- `exps/20260331-200737-flux_dev`
- `exps/20260402-094545-flux_dev`

Important best-model artifacts:

- `exps/best_models/flux.pkl`
- `exps/best_models/sd15/sd15-best.pkl`
- `exps/best_models/sd3-512/sd3-512-best.pkl`
- `exps/best_models/sd3-1024/sd3-1024-best.pkl`

Important sample dirs already referenced heavily:

- `samples/flux_epd_step002800_coco10k`
- `samples/flux_baseline_ipndm_24_coco10k`
- various FLUX baseline dirs under `samples/flux_baseline_*`

## 10. Recommended Startup Checklist for a New Session

1. Read `CONTEXT.md`.
2. Read this handoff document.
3. Check `git status --short`.
4. Decide which protocol you are using:
   - formal clean FID (`score_fid_dir.py`)
   - legacy NPZ FID (`score_fid_npz_dir.py`)
5. Before recursive scoring, clean `.ipynb_checkpoints`.
6. For SD1.5 work, verify `import taming` from outside any source directory.
7. For FLUX training, remember the intended RL semantics are the historical intermediate mixed-prompt state, not standard RLOO.

## 11. Suggested Short Context Snippet for Future Sessions

If you need a smaller seed prompt for a fresh Codex chat, use something like:

> RLEPD is a solver-RL repo for SD1.5 / SD3 / FLUX. Current FLUX formal training intentionally uses the historical intermediate algorithm state between commits `3c58295` and `29987d4`: prompt-major rollout but legacy mixed-prompt PPO baseline. Formal FID is `score_fid_dir.py` = clean-fid on fixed COCO-10k (`src/prompts/coco10k.csv` + `src/coco10k_real_val2014`, clean @256). Legacy auxiliary FID is `score_fid_npz_dir.py` against `src/ms_coco-512x512.npz`; do not mix with formal FID. `benchmark_flux_solver_latency.py` is the main latency tool. SD1.5 currently has a `taming-transformers` local-path issue because the repo prefers `src/taming-transformers`, and that directory is empty. Clean `.ipynb_checkpoints` before recursive scoring. Check `SESSION_HANDOFF_20260417.md`, `CONTEXT.md`, and `README.md` first.
