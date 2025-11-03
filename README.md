RLEPD — PPO Fine-Tuning for EPD Predictor Tables
================================================

RLEPD couples RL with the EPD solver’s parameter-table learning: in `training/ppo/policy.py` a PPO policy outputs per-step Dirichlet concentrations for position segments (`alpha_pos`) and gradient weights (`alpha_weight`) to shape intermediate locations and weights; `training/ppo/cold_start.py` provides a baseline table, and the policy learns residuals in log-concentration space to match the baseline at init while enabling stable refinement; training is orchestrated by `training/ppo/ppo_trainer.py` and `training/ppo/rl_runner.py`, with rewards from `training/ppo/reward_hps.py` (HPS) or external scorers under `reward_models/` and `scripts/score_*.py`; configs and launches use `training/ppo/cfgs/*.yaml`, `launch_rl.sh`, and `launch_baseline.sh`, and `export_epd_predictor.py` exports the learned table for inference. The aim is to adapt EPD’s intermediate sampling across steps to boost stability and generative quality at fixed NFE.

---
