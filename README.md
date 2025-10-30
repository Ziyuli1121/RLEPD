RLEPD — PPO Fine-Tuning for EPD Predictor Tables
================================================

RLEPD fuses two ideas:

1. **EPD (Ensemble Parallel Directions)** &mdash; distilled predictor tables that accelerate diffusion sampling.
2. **TPDM** &mdash; a PPO + RLOO reinforcement-learning framework that optimizes diffusion policies using human preference rewards (HPS).

The objective is to optimize EPD predictor tables directly via RL so that downstream image quality (measured by HPSv2.1, FID, etc.) surpasses the distilled baseline. All RL-specific components live under `training/ppo/`, leaving the original EPD code untouched.

---

Baseline & Environment
----------------------

The repository includes a distilled MS-COCO predictor (`exps/00036-.../network-snapshot-000005.pkl`). Its baseline metrics:

- **NFE 36, 30k prompts**: FID 12.0295  
- **Average HPSv2.1 (first 30 prompts)**: 0.2568

To reproduce the training environment (assuming the original `edm` conda env):

```bash
conda activate edm
pip install omegaconf
conda install lightning -c conda-forge
pip install git+https://github.com/openai/CLIP.git
pip install transformers taming-transformers kornia
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers

cd RLEPD
git clone https://github.com/tgxs002/HPSv2.git
pip install -e HPSv2
python download_hpsv2_weights.py --version v2.1

echo "/work/nvme/betk/zli42/RLEPD/src/taming-transformers" \
  > /work/hdd/becj/zli42/zli42_envs/edm/lib/python3.9/site-packages/taming_local.pth
```

Required assets:

- Stable Diffusion v1.5 weights: `src/ms_coco/v1-5-pruned-emaonly.ckpt`
- HPSv2.1 weights: `weights/HPS_v2.1_compressed.pt`
- MS-COCO prompts: `src/prompts/MS-COCO_val2014_30k_captions.csv`


---

Tests
-----

### Unit Tests (CPU-friendly)

```bash
python -m training.ppo.tests.test_policy
python -m training.ppo.tests.test_rl_runner
python -m training.ppo.tests.test_reward_hps
python -m training.ppo.tests.test_ppo_trainer           # stubbed environment
```

### Integration Tests (GPU + assets)

```bash
EPD_INTEGRATION_TEST=1 python -m training.ppo.tests.test_rl_runner
EPD_INTEGRATION_TEST=1 python -m training.ppo.tests.test_reward_hps
EPD_INTEGRATION_TEST=1 python -m training.ppo.tests.test_ppo_trainer
```

The PPO integration test loads the cold-start table, generates MS-COCO samples through the EPD solver + Stable Diffusion, scores them with HPSv2.1, runs a PPO update, and saves preview images to `training/ppo/tests/integration_outputs/ppo_trainer_integration_sample_*.png`.

---

Training (Stage 7)
------------------

Stage 7 introduces YAML-based configuration, CLI overrides, and standardized run directories.

1. **Dry-run the configuration**
   ```bash
   python -m training.ppo.launch --dry-run
   python -m training.ppo.launch --config training/ppo/cfgs/sd15_base.yaml \
       --override ppo.steps=5 --dry-run
   ```
   Dry-run prints the merged config, validates file paths, and reports `num_steps` / `num_points` inferred from the predictor snapshot without performing any training.

2. **Start training**
   ```bash
   bash launch_rl.sh                                 # default config
   bash launch_rl.sh --override ppo.steps=50         # override specific fields
   python -m training.ppo.launch \
       --config training/ppo/cfgs/sd15_base.yaml \
       --run-name my_experiment
   ```

   - Outputs are written to `exps/<timestamp>-<run_name>/` with subfolders for configs, logs, checkpoints, and samples.  
   - Metrics are streamed to `logs/metrics.jsonl`; console summaries appear every `logging.log_interval` steps.  
   - `ppo.steps` controls how many PPO iterations to run in this session.

3. **Configuration structure**
   - `run`: output root, run name, RNG seed.  
   - `data`: predictor snapshot path, optional custom prompt CSV.  
   - `model`: dataset/guidance hypers (validated against the snapshot).  
   - `reward`: HPS weights, batch size, AMP toggle, cache dir.  
   - `ppo`: rollout batch size, RLOO `k`, learning rate, clip/KL/entropy coefficients, number of steps, Dirichlet concentration.  
   - `logging`: console/log frequencies.  

   See `training/ppo/cfgs/sd15_base.yaml` for the default template; additional overrides can be layered through `--override`.

---

Next Steps
----------

Per `guide2.md`, upcoming stages include:

- **Stage 8**: exporting PPO-trained tables, replay via `sample.py`.  
- **Stage 9–10**: full training runs, logging/monitoring, multi-GPU support.  
- **Stage 11–12**: large-scale evaluation (FID/HPS), ablation studies, feature extensions (`scale_*`, conditional embeddings).  
- **Stage 13**: documentation, cleanup, and final delivery.

---

References
----------

- [guide2.md](guide2.md) — staged development plan (0–13)  
- Original EPD paper & code (`solvers.py`, `train.py`, `sample.py`)  
- TPDM project — reference PPO + RLOO diffusion RL  
- HPSv2.1 — human preference scoring model  

Stage 7 consolidates configuration and launch mechanics so that future experiments (Stage 8 onward) can focus on export, replay, and large-scale training. The RL pipeline still adheres to TPDM’s paradigm while remaining fully compatible with EPD’s solver interface.
