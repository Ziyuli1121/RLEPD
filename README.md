RLEPD
================================================

RLEPD combines two lines of work:

1. **EPD (Ensemble Parallel Directions)** — distilled predictor tables that accelerate diffusion sampling.
2. **TPDM** — a PPO + RLOO reinforcement learning framework that optimizes diffusion policies using human preference rewards.

The goal is to bring TPDM’s reinforcement-learning paradigm into the EPD solver so that the predictor table itself can be optimized for downstream quality metrics (e.g., HPSv2.1). The legacy EPD code remains untouched; all RL components live under `training/ppo/`.


