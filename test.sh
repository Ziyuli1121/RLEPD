# ipndm
python sample_sd3_baseline.py --sampler ipndm --resolution 512 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file src/prompts/test.txt \
  --seeds "0-9" --batch 2 \
  --num-steps 29 --max-order 3 \
  --outdir ./test_samples/ipndm_28step_28nfe

python sample_sd3_baseline.py --sampler ipndm --resolution 512 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file src/prompts/test.txt \
  --seeds "0-9" --batch 2 \
  --num-steps 17 --max-order 3 \
  --outdir ./test_samples/ipndm_16step_16nfe

# dpm
python sample_sd3_baseline.py --sampler dpm2 --resolution 512 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file src/prompts/test.txt \
  --seeds "0-9" --batch 2 \
  --num-steps 15 \
  --outdir ./test_samples/dpm_14step_28nfe

python sample_sd3_baseline.py --sampler dpm2 --resolution 512 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file src/prompts/test.txt \
  --seeds "0-9" --batch 2 \
  --num-steps 9 \
  --outdir ./test_samples/dpm_8step_16nfe

# edm
python sample_sd3_baseline.py --sampler edm --resolution 512 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file src/prompts/test.txt \
  --seeds "0-9" --batch 2 \
  --num-steps 14 \
  --outdir ./test_samples/edm_14step_28nfe

python sample_sd3_baseline.py --sampler edm --resolution 512 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file src/prompts/test.txt \
  --seeds "0-9" --batch 2 \
  --num-steps 8 \
  --outdir ./test_samples/edm_8step_16nfe

# default
python sample_sd3_baseline.py --sampler sd3 --resolution 512 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file src/prompts/test.txt \
  --seeds "0-9" --batch 2 \
  --num-steps 28 \
  --outdir ./test_samples/default_28step_28nfe

python sample_sd3_baseline.py --sampler sd3 --resolution 512 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt-file src/prompts/test.txt \
  --seeds "0-9" --batch 2 \
  --num-steps 16 \
  --outdir ./test_samples/default_16step_16nfe

# epd
python sample_sd3.py \
  --predictor exps/fake-sd3-11-512-compare/network-snapshot-000005.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds "0-9" \
  --max-batch-size 1 \
  --outdir samples/test_compare
