RLEPD — PPO Fine-Tuning for EPD Predictor Tables
================================================

RLEPD fuses two ideas:

1. **EPD (Ensemble Parallel Directions)** &mdash; distilled predictor tables that accelerate diffusion sampling.
2. **TPDM** &mdash; a PPO + RLOO reinforcement-learning framework that optimizes diffusion policies using human preference rewards (HPS).

The objective is to optimize EPD predictor tables directly via RL so that downstream image quality (measured by HPSv2.1, FID, etc.) surpasses the distilled baseline. All RL-specific components live under `training/ppo/`, leaving the original EPD code untouched.

---
