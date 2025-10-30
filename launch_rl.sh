#!/usr/bin/env bash

# Simple launcher for PPO fine-tuning of EPD predictor tables.
# Usage:
#   bash launch_rl.sh                          # use default config
#   bash launch_rl.sh --config my_cfg.yaml     # specify custom config
#   bash launch_rl.sh --override ppo.steps=50  # override values

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="training/ppo/cfgs/sd15_base.yaml"

python -m training.ppo.launch \
  --config "${DEFAULT_CONFIG}" \
  "$@"
