python benchmark_flux_solver_latency.py \
  --model-id /work/nvme/betk/zli42/RLEPD/weights/hf_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21 \
  --prompt-file src/prompts/test.txt \
  --seeds 0-99 \
  --batch-size 1 \
  --baseline-solvers euler,edm,dpm2,ipndm \
  --euler-steps 16 \
  --edm-steps 8 \
  --dpm2-steps 9 \
  --ipndm-steps 17 \
  --outdir results/latency_baseline_16

python benchmark_flux_solver_latency.py \
  --model-id /work/nvme/betk/zli42/RLEPD/weights/hf_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21 \
  --prompt-file src/prompts/test.txt \
  --seeds 0-99 \
  --batch-size 1 \
  --baseline-solvers euler,edm,dpm2,ipndm \
  --euler-steps 20 \
  --edm-steps 10 \
  --dpm2-steps 11 \
  --ipndm-steps 21 \
  --outdir results/latency_baseline_20

python benchmark_flux_solver_latency.py \
  --model-id /work/nvme/betk/zli42/RLEPD/weights/hf_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21 \
  --prompt-file src/prompts/test.txt \
  --seeds 0-99 \
  --batch-size 1 \
  --baseline-solvers euler,edm,dpm2,ipndm \
  --euler-steps 24 \
  --edm-steps 12 \
  --dpm2-steps 13 \
  --ipndm-steps 25 \
  --outdir results/latency_baseline_24