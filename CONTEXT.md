# RLEPD Context

This file is a durable project context note for future coding/research agents working on this repository.

Goal: make the repo easier to re-enter without re-deriving the pipeline, the algorithm intent, the main code paths, and the current engineering constraints from scratch.

Last refreshed: 2026-04-12

## 1. Project Summary

RLEPD is a reinforcement-learning pipeline for tuning an EPD-Solver table used in text-to-image diffusion sampling.

The fixed intended pipeline is:

1. `fake_train.py` builds a cold-start `EPD_predictor` parameter table and fixes the planned inference step count.
2. Choose a backbone: `sd1.5`, `sd3-512`, `sd3-1024`, or `flux`.
3. Run RDPO/PPO training over the EPD table with `training/ppo/launch.py`.
4. Export the trained policy mean back into predictor format with `training/ppo/export_epd_predictor.py`.
5. Sample images with `sample.py`, `sample_sd3.py`, `sample_flux.py`, or `sample_flux_baseline.py`.
6. Evaluate images with `clip`, `hps`, `aesthetic`, `pickscore`, `imagereward`, and `mps`.

The repo has been cleaned to make this pipeline statically closed without changing the public CLI contract or artifact formats.

## 2. High-Level Idea

The project does not directly RL-train a full diffusion model. It RL-trains the solver table that controls how EPD sampling moves between coarse diffusion steps.

The mental model is:

- A traditional `EPD_predictor` stores per-step table parameters:
  - `r`: intermediate positions inside a coarse step.
  - `weight`: simplex weights for combining multiple candidate directions.
  - optional `scale_dir` and `scale_time`: multiplicative correction factors.
- The actual EPD solver in `solvers.py` consumes those tables during sampling.
- RDPO replaces direct table regression with a stochastic policy:
  - positions are represented as `K+1` simplex segments, then cumulative-summed back to monotonic `r`.
  - weights are represented directly on a simplex of size `K`.
  - optional `scale_dir` and `scale_time` are modeled with log-normal factors.
- The policy is initialized so that its mean exactly matches the cold-start EPD table.
- PPO then learns residuals on top of that cold-start table.
- After training, the policy mean is exported back to the legacy predictor format so all existing sampling code can reuse it.

In short: cold-start table -> Dirichlet/log-normal policy around that table -> PPO optimization -> export back to old predictor -> reuse legacy sampling/eval pipeline.

## 3. Supported Backbones and Scope

Officially supported backbone workflows in the current repo:

- `sd1.5`
- `sd3-512`
- `sd3-1024`
- `flux` (`FLUX.1-dev`, 1024-only in this checkout)

Important scope boundary:

- This trimmed repository currently treats `ms_coco` as the supported dataset/task path.
- Single-node GPU is the intended runtime.
- The cleaned code supports static closure of the full pipeline.
- As of 2026-03-29, a real smoke run has exercised cold-start, train, export, and sample for `sd1.5`, `sd3-512`, and `sd3-1024`.
- `flux` is now wired into the same pipeline, but its runtime still depends on an environment whose installed `diffusers` exposes `FluxPipeline`.
- Not every metric/backbone combination or long-running training regime has been proven.

Important current project choice:

- As of 2026-03-31, the intended algorithm for formal `FLUX.1-dev` EPD-Solver training in this checkout is the historical intermediate state between commits:
  - `3c58295b7ab7b564ebc3a3337680704b8eec0a70`
  - `29987d46524d65637f8b9f1daddf75922943fb61`
- That means:
  - rollout prompt layout is `prompt-major`
  - PPO advantage computation still uses the legacy mixed-prompt leave-one-out baseline
- Future agents should not silently "fix" this back to standard same-prompt RLOO unless the user explicitly asks for that change.

## 4. Fixed End-to-End Pipeline

| Stage | Purpose | Main entrypoint | Main output |
| --- | --- | --- | --- |
| 1 | Build cold-start EPD table | `fake_train.py` | `network-snapshot-*.pkl` |
| 2 | PPO / RDPO training | `python -m training.ppo.launch` | `checkpoints/policy-step*.pt` |
| 3 | Export predictor | `python -m training.ppo.export_epd_predictor` | `export/network-snapshot-export-step*.pkl` |
| 4 | Sample images | `sample.py` / `sample_sd3.py` / `sample_flux.py` / `sample_flux_baseline.py` | `samples/.../*.png` |
| 5 | Score quality | `training/ppo/scripts/score_*.py` | `results/*.json` |

### Stage 1: Cold Start

Entrypoint: `fake_train.py`

Responsibilities:

- Create a synthetic/default `EPD_predictor`.
- Initialize `r` and `weight` tables from simple priors.
- Persist downstream metadata into the predictor object and `training_options.json`.
- Fix solver-level settings such as:
  - `num_steps`
  - `num_points`
  - `guidance_rate`
  - `schedule_type`
  - `backend`
  - `backend_config`
  - `resolution`
  - `sigma_min`
  - `sigma_max`
  - `flowmatch_mu`
  - `flowmatch_shift`

Why this matters: later RL/export/sample stages now rely on predictor metadata as the source of truth whenever possible.

### Stage 2: PPO / RDPO Training

Entrypoint: `training/ppo/launch.py`

Responsibilities:

- Load and validate YAML config.
- Load cold-start predictor table from the fake snapshot.
- Convert cold-start tables into Dirichlet initialization.
- Construct `EPDParamPolicy`.
- Load the diffusion backbone through `sample.create_model_backend`.
- Build reward module.
- Build `EPDRolloutRunner`.
- Run PPO updates with `PPOTrainer`.
- Save resolved config, metrics, and policy checkpoints into a run directory.

Important default training behavior:

- official configs currently use `reward.type=multi`
- but the shipped backbone configs set `hps=1.0` and other metric weights to `0.0`
- so the default training behavior is effectively HPS-only, even though the multi-reward framework is already wired in
- the currently intended FLUX training semantics are not standard same-prompt RLOO:
  - `rl_runner.py` emits prompt-major batches
  - `ppo_trainer.py` still computes advantages with the historical mixed-prompt baseline (`view(k, num_groups)` with leave-one-out over `dim=0`)
  - example: with `rollout_batch_size=8` and `rloo_k=4`, rollout generation layout is `2` prompts with `4` images each, but PPO advantage groups are still mixed across those prompt blocks

Important current configs:

- `training/ppo/cfgs/sd15_base.yaml`
- `training/ppo/cfgs/sd15_k5.yaml`
- `training/ppo/cfgs/sd15_k20.yaml`
- `training/ppo/cfgs/sd15_k50.yaml`
- `training/ppo/cfgs/sd3_512.yaml`
- `training/ppo/cfgs/sd3_1024.yaml`
- `training/ppo/cfgs/flux_dev.yaml`

### Stage 3: Export

Entrypoint: `training/ppo/export_epd_predictor.py`

Responsibilities:

- Load the Stage 2 run directory and resolved config.
- Find the requested or latest `policy-step*.pt`.
- Reconstruct the policy architecture.
- Compute policy mean tables.
- Sanitize tables if needed.
- Convert them back to the standard `EPD_predictor` weight structure.
- Write export artifacts:
  - predictor snapshot
  - export `training_options.json`
  - export manifest

Key design point: export is the bridge that keeps the old sampling code working unchanged.

Current export artifact names:

- `network-snapshot-export-stepXXXXXX.pkl`
- `training_options-export-stepXXXXXX.json`
- `export-manifest-stepXXXXXX.json`

### Stage 4: Sampling

Entrypoints:

- `sample.py` for SD1.5 / legacy-style predictor replay
- `sample_sd3.py` for SD3 EPD replay
- `sample_flux.py` for FLUX.1-dev EPD replay
- `sample_baseline.py` for SD1.5 baseline samplers
- `sample_sd3_baseline.py` for SD3 baseline samplers
- `sample_flux_baseline.py` for FLUX baseline samplers
- `compare_flux_euler.py` for internal FLUX Euler equivalence/debug comparisons

Responsibilities:

- Resolve predictor path from either:
  - direct `.pkl`
  - RL run directory
  - export directory
- Load prompts from text or CSV.
- Load the appropriate backbone/backend.
- Rebuild solver settings from predictor metadata.
- Run the solver and decode images.
- Write images into the standard `samples/` tree.

Important FLUX-specific sampling note:

- `sample_flux.py` is the FLUX EPD replay path.
- `sample_flux_baseline.py` is the FLUX baseline path.
- In this repo, "FLUX default Euler" means the official diffusers `FluxPipeline + FlowMatchEulerDiscreteScheduler`.
- In this repo, the closest project-side Euler-style comparator is `ddim` with `schedule_type=flowmatch`, not `edm`.
- Official FLUX Euler and `ddim(flowmatch)` should be treated as different baselines; use `compare_flux_euler.py` when that distinction matters.
- FLUX baseline currently uses embedded `guidance_scale`; true CFG / negative prompt branches are intentionally not wired into the RLEPD FLUX path.
- The official FLUX baseline path should treat `output_type="pt"` as an already postprocessed `[0, 1]` tensor.
- The official FLUX baseline path should not be wrapped in the default CUDA autocast context; this checkout loads FLUX in BF16 and mismatched autocast can corrupt outputs.

### Stage 5: Evaluation

Entrypoints:

- `training/ppo/scripts/score_clip.py`
- `training/ppo/scripts/score_hps.py`
- `training/ppo/scripts/score_aesthetic.py`
- `training/ppo/scripts/score_pick.py`
- `training/ppo/scripts/score_imagereward.py`
- `training/ppo/scripts/score_mps.py`
- `training/ppo/scripts/score_fid_dir.py`
- `training/ppo/scripts/score_fid_npz_dir.py`

Responsibilities:

- Load image directory plus prompt file.
- Resolve local weights first.
- Fall back to remote downloads only if local weights are absent.
- Emit JSON score summaries under `results/`.
- For formal FLUX fidelity evaluation, `score_fid_dir.py` computes:
  - `FID-10k (fixed COCO subset, clean preprocessing, native 1024 generation)`
  - fake images from a generated sample directory
  - real reference from `src/coco10k_real_val2014`
  - protocol source of truth from `src/prompts/coco10k.csv`
- `score_fid_dir.py` uses `clean-fid` and caches custom real stats under `results/fid_cache/`.
- `score_fid_npz_dir.py` computes a separate legacy auxiliary metric:
  - Inception-FID against precomputed reference stats from `src/ms_coco-512x512.npz`
  - this is not the same protocol as `score_fid_dir.py`
  - do not mix or directly compare those two FID numbers in one table without explicit labeling
- `test_flux.sh` now generates a prompt subset file aligned to the sampled seeds before scoring, so its default `1 image` FLUX eval path is valid.
- FLUX formal eval defaults to full metrics on the EPD sample and generation-only checks for baseline solver sweeps.

## 5. Core Algorithm Objects

### `training/networks.py`

`EPD_predictor` is the legacy/persistent table module.

What it stores:

- `r_params` -> sigmoid -> `r` in `(0, 1)`
- `weight_s` -> softmax -> simplex weights
- `scale_dir_params`
- `scale_time_params`

It also stores solver metadata that downstream tooling now depends on:

- `dataset_name`
- `img_resolution`
- `num_steps`
- `sampler_stu`
- `sampler_tea`
- `guidance_type`
- `guidance_rate`
- `schedule_type`
- `schedule_rho`
- `backend`
- `backend_config`

Shape invariant:

- solver tables are global per-step tables of shape `(num_steps - 1, num_points)`
- `num_steps` includes both endpoints
- `r` rows must remain strictly ordered inside `(0, 1)`
- `weight` rows must stay positive and sum to `1`

### `solvers.py`

This is the actual sampling logic. The EPD update rule lives here, not in PPO code.

Future agents should treat the solver as the behavior-defining runtime and the policy as the generator of its table parameters.

One important runtime detail:

- the solver computes midpoint times using the learned `r` table
- it evaluates midpoint directions/velocities
- it combines them using `weight * scale_dir`

### `training/ppo/cold_start.py`

This file converts an old predictor table into RL-friendly parameterization:

- `positions -> segments`
- normalize simplex values
- construct Dirichlet concentration tensors with a configurable concentration value
- preserve round-trip compatibility back to table means

This is the mathematical bridge from old deterministic table to new stochastic policy.

### `training/ppo/policy.py`

`EPDParamPolicy` is the main RDPO policy.

Important details:

- It predicts per-step Dirichlet parameters for position segments and weights.
- It uses cold-start concentration tensors as frozen buffers.
- Learnable outputs are residuals in log-concentration space.
- At initialization, the policy mean matches the cold-start table exactly.
- Optional `scale_dir` / `scale_time` are modeled independently with log-normal heads.
- The current policy is indexed by coarse step only; it is not prompt-conditioned in the present implementation.

This is the main meaning of "Residual Dirichlet Policy Optimization" in the current codebase.

This detail matters for baseline design:

- because the policy is not prompt-conditioned, the historical mixed-prompt baseline can empirically behave differently from standard same-prompt RLOO
- future agents should not assume that "same-prompt RLOO" is automatically the intended or best objective for current FLUX training

### `training/ppo/rl_runner.py`

`EPDRolloutRunner` is the bridge between PPO and legacy sampling:

- samples whole EPD tables from the policy
- adapts them into a predictor-like module
- prepares prompts, seeds, and latents
- calls the old EPD solver
- returns images and rollout metadata for PPO

This file is where "RL policy world" meets "existing solver world".

Important current semantic note:

- The current desired FLUX training mode uses the historical intermediate rollout layout:
  - prompts are emitted in prompt-major order
  - each prompt is repeated contiguously `rloo_k` times before advancing
- For `rollout_batch_size=8, rloo_k=4`, the generated prompt layout is:
  - `[p0, p0, p0, p0, p1, p1, p1, p1]`
- This rollout layout should not be confused with the PPO baseline semantics; see `ppo_trainer.py` below.

### `training/ppo/ppo_trainer.py`

`PPOTrainer` computes leave-one-out advantages and runs PPO updates.

Important current semantic note:

- As of 2026-03-31, the intended FLUX training path in this checkout uses the legacy mixed-prompt baseline, not standard prompt-major RLOO.
- The current advantage code reshapes rewards as:
  - `rewards.view(k, num_groups)`
  - and computes leave-one-out baselines over `dim=0`
- Combined with the prompt-major rollout layout above, `rollout_batch_size=8, rloo_k=4` gives:
  - generated images for `2` prompts, `4` images each
  - but PPO advantage groups are column-wise mixed:
    - `{r0, r2, r4, r6}`
    - `{r1, r3, r5, r7}`
- This is intentionally the historical intermediate behavior between `3c58295` and `29987d4`.

### `training/ppo/export_epd_predictor.py`

This file maps policy mean tables back into `EPD_predictor` snapshot format.

If export breaks, the whole legacy sample/test chain breaks even if PPO training itself is correct.

## 6. Important Engineering Helpers

### `training/ppo/pipeline_utils.py`

This is the new shared Python helper layer for common pipeline behavior.

It currently centralizes:

- repo roots and default paths
- Python-version-aware `diffusers` bootstrap
- local vendored `HPSv2` bootstrap
- local vendored `taming-transformers` bootstrap
- prompt loading from `.txt` and `.csv`
- local weight alias resolution
- predictor path resolution from:
  - `.pkl`
  - run dir
  - export dir
  - numeric experiment id

Future agents should modify this file first when the problem is "path resolution / prompt resolution / local weight / predictor lookup drift".

### `scripts/pipeline_common.sh`

This is the shell-side equivalent helper layer.

It centralizes:

- repo root constants
- default prompt paths
- existence checks
- latest run/checkpoint/export lookup
- `run_export_predictor`
- `score_all_metrics`

Future shell-script fixes should prefer touching this file instead of duplicating more path logic.

## 7. Recommended Entrypoints

Current top-level scripts after cleanup:

- `TEST.sh`
  - the most complete low-cost smoke path for this repo
  - builds fresh cold-start snapshots, runs 10 PPO steps, exports predictors, samples images, and runs representative evals
- `train.sh`
  - example pipeline for fake train + PPO + export
- `launch.sh`
  - end-to-end launch examples
- `sample.sh`
  - SD1.5 sampling examples
- `sample512.sh`
  - SD3-512 sampling/eval examples
- `sample1024.sh`
  - SD3-1024 sampling/eval examples
- `test_regression.sh`
  - smaller export/sample/score regression path against existing runs
- `test_15.sh`
- `test_512.sh`
- `test_1024.sh`
  - experiment-oriented scripts that now follow `export -> sample -> score` structure

Recommended smoke entrypoint:

```bash
./TEST.sh
SD3_MODEL_PATH=/path/to/local/sd3_snapshot ./TEST.sh
ENABLE_PICKSCORE_DOWNLOAD=1 ./TEST.sh
```

Recommended lighter regression entrypoint:

```bash
BACKBONE=sd15 bash test_regression.sh
BACKBONE=sd3-512 bash test_regression.sh
BACKBONE=sd3-1024 bash test_regression.sh
```

## 8. Directory and Artifact Conventions

Important directories:

- `exps/f15`
- `exps/f512`
- `exps/f1024`
  - fake/cold-start snapshots
- `exps/<timestamp>-<run_name>/`
  - PPO run directories
- `samples/`
  - generated images
- `results/`
  - metric JSONs
- `weights/`
  - local metric/model weights

Important artifact contracts that should not be broken:

- cold-start snapshot: `network-snapshot-*.pkl`
- RL checkpoint: `checkpoints/policy-stepXXXXXX.pt`
- export predictor: `export/network-snapshot-export-stepXXXXXX.pkl`
- resolved config: `configs/resolved_config.yaml`
- metrics log: `logs/metrics.jsonl`
- sample outputs: `samples/<name>/<bucket>/<seed>.png`
- eval outputs: `results/<prefix>_<metric>.json`

## 9. Prompt and Weight Conventions

Default prompt files:

- text prompts: `src/prompts/test.txt`
- CSV prompts: `src/prompts/MS-COCO_val2014_30k_captions.csv`

Current prompt behavior:

- Python helpers support both `.txt` and `.csv`.
- `sample.py` can use prompt files directly.
- `sample_sd3.py` can use prompt files directly.
- RL prompt streaming in `rl_runner.py` uses the CSV path by default.

Important nuance:

- `sample.py` maps prompts using seed indices when sampling from prompt files without a fixed `--prompt`.
- `sample_sd3.py` repeats/cycles the prompt list to match the requested seed count.
- RL rollout defaults to the MS-COCO CSV path, not `test.txt`, when `prompt_csv` is unset.
- `TEST.sh` uses tiny text prompt files only for sampling/eval smoke checks; training still uses the default MS-COCO CSV path unless explicitly overridden.
- Future agents should not silently assume those two sampling paths are identical in prompt indexing semantics.

Weight aliases currently resolved locally first:

- HPS -> `weights/HPS_v2.1_compressed.pt`
- ImageReward -> `weights/ImageReward.pt` or `weights/ImageReward-v1.0.pt`
- PickScore -> `weights/PickScore_v1`
- CLIP cache -> `weights/clip`
- Aesthetic -> `weights/sac+logos+ava1-l14-linearMSE.pth`
- MPS -> `weights/MPS_overall_checkpoint.pth`

Practical weight note:

- `PickScore` is not assumed to be locally bundled.
- `TEST.sh` skips PickScore by default if `weights/PickScore_v1` is absent.
- Set `ENABLE_PICKSCORE_DOWNLOAD=1` only if a future run intentionally wants to exercise the Hugging Face download path.

## 10. Environment and Dependency Assumptions

Current user environment observed during validation:

- conda env: `epd`
- Python: `3.9.25`
- PyTorch: `2.8.0+cu128`

Repo/runtime assumptions:

- local vendored `HPSv2` is preferred for HPS reward/eval
- local vendored `taming-transformers` is preferred for SD1.5 runtime imports
- local vendored `diffusers` is preferred only when the active Python version can import it safely
- GPU execution is the intended path
- FLUX formal single-node runs should prefer [environment.flux.yml](/work/nvme/betk/zli42/RLEPD/environment.flux.yml), which pins the validated Python 3.9 package family

Important environment caveat:

- The repo's vendored `diffusers` declares `python>=3.10`.
- The current `epd` environment is Python 3.9.
- On Python 3.9, `pipeline_utils.bootstrap_local_diffusers()` now prefers an installed site-packages `diffusers` if available instead of forcing the vendored tree.
- FLUX formal scripts now run a dedicated runtime preflight that checks the known-good Python / torch / diffusers / transformers / hub package family, verifies local/remote model reachability, and confirms that the installed `diffusers` package contains the FLUX pipeline files required by this checkout.
- `TEST.sh` now sources the shared pipeline helpers as well, so the FLUX smoke path uses the same runtime preflight and prompt-alignment helpers as `train.sh` / `test_flux.sh`.
- During the 2026-03-29 smoke run, SD3 training and sampling succeeded in `epd` by using the installed `diffusers` package plus a local cached SD3 snapshot.
- The SD3 model itself is still a gated Hugging Face asset. Real SD3 runtime requires either:
  - a local snapshot passed via `SD3_MODEL_PATH` or auto-discovered from cache
  - or authenticated remote access with `SD3_ALLOW_REMOTE=1`

Do not overclaim runtime compatibility beyond what has actually been tested.

## 11. Current Validation Status

As of 2026-03-29, the following have been verified:

- `python -m py_compile` passed on the modified Python files.
- `bash -n` passed on the modified shell scripts.
- `python -m training.ppo.launch --dry-run` passed in the `epd` environment for:
  - `training/ppo/cfgs/sd15_base.yaml`
  - `training/ppo/cfgs/sd3_512.yaml`
  - `training/ppo/cfgs/sd3_1024.yaml`
- A real end-to-end smoke run completed successfully under:
  - `smoke_tests/20260329-162229`

That smoke run exercised:

- Stage 1 fake cold-start generation for:
  - `sd1.5`
  - `sd3-512`
  - `sd3-1024`
- Stage 2 PPO training for all three backbones:
  - official backbone YAML hyperparameters preserved
  - total PPO steps reduced to `10`
  - save interval reduced to `10`
- Stage 3 predictor export for all three runs:
  - `policy-step000010.pt`
  - `network-snapshot-export-step000010.pkl`
  - `training_options-export-step000010.json`
  - `export-manifest-step000010.json`
- Stage 4 sampling for all three runs:
  - `sd15`: `5` images
  - `sd3_512`: `2` images
  - `sd3_1024`: `1` image
- Stage 5 evaluation:
  - full `clip/hps/aesthetic/imagereward/mps` suite on `sd15`
  - lightweight `hps` eval on `sd3_512` and `sd3_1024`

Observed smoke-run facts:

- The smoke train path preserved official PPO hyperparameters from the YAML files:
  - `sd15`: `rollout_batch_size=8`, `rloo_k=4`, `minibatch_size=4`, `reward.batch_size=4`, `learning_rate=7e-5`
  - `sd3_512`: `rollout_batch_size=16`, `rloo_k=4`, `minibatch_size=4`, `reward.batch_size=4`, `learning_rate=7e-5`
  - `sd3_1024`: `rollout_batch_size=8`, `rloo_k=4`, `minibatch_size=4`, `reward.batch_size=4`, `learning_rate=7e-5`
- `TEST.sh` reduced fake cold-start solver step counts for cost control:
  - `sd15`: `num_steps=4`
  - `sd3_512`: `num_steps=4`
  - `sd3_1024`: `num_steps=3`
- Export manifests for all three runs reported:
  - `export_step=10`
  - `sanitized_rows.reordered=0`
  - `sanitized_rows.adjusted=0`
- The default reward path is still effectively HPS-only:
  - in the smoke metrics, `mixed_reward_mean` tracked the HPS reward because the shipped configs keep other reward weights at `0`

What has not been fully proven inside this context:

- a long-running PPO training cycle beyond smoke scale
- the PickScore code path in runtime, because local `weights/PickScore_v1` was absent during the validated smoke run
- a full six-metric eval suite on `sd3_512` and `sd3_1024`, because `TEST.sh` currently runs lightweight HPS-only eval there
- a full FLUX baseline solver matrix inside `TEST.sh`, because that sweep is now optional behind `FLUX_SOLVER_SWEEP=1`

Future agents should distinguish among:

- static closure / path closure / config closure
- smoke-runtime proof via `TEST.sh`
- long-run experiment proof

## 12. Known Caveats and Non-Obvious Details

### 12.1 Legacy Stage-1 code still exists

Some older code paths are not the current PPO mainline.

Examples:

- `training/loss.py` still contains legacy assumptions such as `predictor.module`.
- Some legacy sampler assertions allow more values than `get_solver_fn` actually supports.

These are legacy distillation-era rough edges, not the main RDPO pipeline.

### 12.2 `launch.py` uses lazy reward imports on purpose

`training/ppo/launch.py` now imports reward modules only after config validation and after `--dry-run` exits.

Reason:

- avoids dry-run hanging or failing because `reward_hps` triggers heavy imports too early

Future agents should preserve this behavior.

### 12.3 `sample.py` still initializes distributed utilities

`sample.py` calls `torch_utils.distributed.init()` and is effectively written for GPU/distributed-friendly execution.

Do not assume it is a clean CPU-only script.

### 12.4 `return_inters` is compatibility-preserved, not a full feature

`sample.py --return_inters` is accepted and no longer crashes, but the current compatibility path does not implement a rich intermediate-saving workflow.

Do not overstate what it does.

### 12.5 Predictor metadata is now part of the pipeline contract

Many resolved settings are intentionally taken from predictor metadata rather than trusting YAML or CLI overrides blindly.

If a future change touches:

- backend
- resolution
- schedule
- flowmatch parameters
- sigma range

then the change must stay coherent across:

- `fake_train.py`
- `launch.py`
- export logic
- sample logic

Also preserve this semantic invariant:

- RDPO currently learns a distribution over solver tables indexed only by coarse diffusion step
- it does not condition the policy on prompt text or image content
- if that assumption is changed later, the rollout interface, policy signature, export semantics, and comparison methodology all need to be re-audited

Also preserve this current intent:

- for `FLUX.1-dev` formal training in this checkout, the user currently wants the historical intermediate mixed-baseline algorithm, not the later same-prompt RLOO revision
- many successful 2025-11 to 2025-12 `sd1.5` / `sd3` experiments were produced during this intermediate algorithm state

### 12.6 `TEST.sh` is the main full smoke script

`TEST.sh` is currently the most important one-command validation path for this repo.

Important characteristics:

- it builds fresh fake snapshots instead of assuming pre-existing runs
- it preserves official PPO hyperparameters from the selected YAMLs
- it only reduces total train length and cold-start solver step counts to keep cost manageable
- it requires local or authenticated access to the gated SD3 model
- it runs full eval on `sd15` and lightweight HPS eval on `sd3` backbones

If a future agent needs one stable end-to-end smoke path, start with `TEST.sh`.

### 12.7 `test_regression.sh` is a lighter existing-run regression script

`test_regression.sh` is still useful, but its role is narrower than `TEST.sh`.

It assumes a suitable run directory already exists, then performs:

- export
- sample
- score

It is not the best first script when the goal is to prove that fresh cold-start generation and fresh PPO training still run.

### 12.8 `train.sh` is an example pipeline, not a perfect full export matrix

`train.sh` now runs the cold-start and PPO examples coherently, but its default auto-export coverage is still selective:

- it auto-exports the latest `sd15_k20` run
- it auto-exports the latest `sd3_512_new` run
- it does not automatically export the `sd3_1024_continue` run by default

If a future task expects symmetric export behavior across all backbones, audit `train.sh` explicitly instead of assuming it already does that.

### 12.9 Current warnings are mostly non-blocking cleanup targets

The validated smoke run still emitted a few warnings that did not block correctness:

- `sample.py` can exit with a distributed cleanup warning about `destroy_process_group()`
- `reward_hps.py` still uses a deprecated `torch.cuda.amp.autocast(...)` form
- the installed `diffusers` path emits `torch_dtype` deprecation warnings
- Pillow image save calls emit a future deprecation warning for the `mode` argument

### 12.10 Current FLUX training objective is historically pinned on purpose

There are three distinct algorithm states in repo history:

1. original mixed-prompt:
   - rollout sampled prompts sample-by-sample
   - PPO used `view(k, num_groups)` with leave-one-out over `dim=0`
2. intermediate state:
   - rollout changed to prompt-major contiguous repeats in `3c58295`
   - PPO still used the old mixed-prompt baseline
3. later standard RLOO:
   - PPO changed to `view(num_prompts, k)` with leave-one-out over `dim=1` in `29987d4`

Current intended FLUX training in this checkout uses state 2.

Why this note exists:

- multiple strong 2025-11/2025-12 `sd1.5` and `sd3` experiments were produced during state 2
- the user currently wants to carry that same algorithm family over to full `FLUX.1-dev` EPD-Solver training
- future agents should therefore treat state 2 as intentional project behavior, not as an accidental half-migration that should be cleaned up automatically

Future agents can clean these up, but they should not misclassify them as evidence that the pipeline failed.

## 13. Code Map

Algorithm and solver:

- `training/networks.py`
- `solvers.py`
- `training/ppo/cold_start.py`
- `training/ppo/policy.py`

PPO / RL pipeline:

- `training/ppo/config.py`
- `training/ppo/launch.py`
- `training/ppo/rl_runner.py`
- `training/ppo/ppo_trainer.py`
- `training/ppo/reward_hps.py`
- `training/ppo/reward_multi.py`
- `training/ppo/export_epd_predictor.py`

Shared pipeline helpers:

- `training/ppo/pipeline_utils.py`
- `scripts/pipeline_common.sh`

Sampling:

- `sample.py`
- `sample_sd3.py`
- `sample_flux.py`
- `sample_baseline.py`
- `sample_sd3_baseline.py`
- `sample_flux_baseline.py`
- `models/backends/sd3_diffusers_backend.py`
- `models/backends/flux_diffusers_backend.py`

Evaluation:

- `training/ppo/scripts/score_clip.py`
- `training/ppo/scripts/score_hps.py`
- `training/ppo/scripts/score_aesthetic.py`
- `training/ppo/scripts/score_pick.py`
- `training/ppo/scripts/score_imagereward.py`
- `training/ppo/scripts/score_mps.py`

Operational scripts:

- `TEST.sh`
- `train.sh`
- `launch.sh`
- `sample.sh`
- `sample512.sh`
- `sample1024.sh`
- `test_regression.sh`
- `test_15.sh`
- `test_512.sh`
- `test_1024.sh`

## 14. Guidance for Future Agents

If you are a future coding/research agent entering this repo:

1. Read this file first.
2. If the task is training-related, inspect `training/ppo/launch.py`, `training/ppo/config.py`, and `training/ppo/rl_runner.py`.
3. If the task is sample/export-related, inspect `sample.py`, `sample_sd3.py`, and `training/ppo/export_epd_predictor.py`.
4. If the task is path/config drift, inspect `training/ppo/pipeline_utils.py` and `scripts/pipeline_common.sh` before editing scattered scripts.
5. Preserve public CLI names and artifact formats unless the user explicitly asks for a breaking change.
6. Before claiming "the full pipeline runs", distinguish among dry-run proof, `TEST.sh` smoke-runtime proof, and broader long-run experiment proof.
7. If adding a new backbone, update all of these together:
   - cold-start metadata generation
   - YAML config
   - train launch path
   - export behavior
   - sample entrypoint
   - regression/test script
   - weight/prompt/path resolution if needed

## 15. Practical Invariants to Preserve

These are easy places to accidentally introduce regressions:

- Do not break predictor `.pkl` compatibility.
- Do not break `policy-step*.pt` checkpoint naming.
- Do not move export outputs out of `run_dir/export/`.
- Do not re-fragment prompt/weight/predictor resolution logic; keep it centralized.
- Do not reintroduce stale hard-coded experiment directories into scripts.
- Do not remove lazy reward imports from `launch.py`.
- Do not silently change prompt indexing semantics without explicitly auditing both `sample.py` and `sample_sd3.py`.

## 16. One-Sentence Project Definition

RLEPD is a solver-table RL system: it cold-starts an EPD solver table, optimizes that table with a residual Dirichlet PPO policy for SD1.5 or SD3 text-to-image sampling, exports the learned mean back into legacy predictor format, then evaluates image quality with a fixed multi-metric scoring suite.
