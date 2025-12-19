################################ sd1.5 ################################
# r0:0.5 | w0:1.00000
python fake_train.py \
  --num-steps 11 \
  --num-points 1 \
  --outdir exps/latency/sd15_k1_nfe20 \
  --guidance-rate 1.0 \
  --r-base 0.5 \
  --r-epsilon 0.0001

# r0:0.33500 r1:0.66500 | w0:0.50000 w1:0.50000
python fake_train.py \
  --num-steps 11 \
  --num-points 2 \
  --outdir exps/latency/sd15_k2_nfe20 \
  --guidance-rate 1.0 \
  --r-base 0.5 \
  --r-epsilon 0.33

# r0:0.17000 r1:0.50000 r2:0.83000 | w0:0.33333 w1:0.33333 w2:0.33333
python fake_train.py \
  --num-steps 11 \
  --num-points 3 \
  --outdir exps/latency/sd15_k3_nfe20 \
  --guidance-rate 1.0 \
  --r-base 0.5 \
  --r-epsilon 0.33

################################ sd3-512 ################################
# r0:0.5 | w0:1.00000
python fake_train.py \
  --outdir exps/latency/sd3-512_k1_nfe20 \
  --num-steps 11 \
  --num-points 1 \
  --guidance-rate 4.5 \
  --schedule-type flowmatch \
  --backend sd3 \
  --resolution 512 \
  --backend-options '{"model_name_or_path":"stabilityai/stable-diffusion-3-medium-diffusers"}' \
  --r-base 0.5 \
  --r-epsilon 0.0001

# r0:0.33500 r1:0.66500 | w0:0.50000 w1:0.50000
python fake_train.py \
  --outdir exps/latency/sd3-512_k2_nfe20 \
  --num-steps 11 \
  --num-points 2 \
  --guidance-rate 4.5 \
  --schedule-type flowmatch \
  --backend sd3 \
  --resolution 512 \
  --backend-options '{"model_name_or_path":"stabilityai/stable-diffusion-3-medium-diffusers"}' \
  --r-base 0.5 \
  --r-epsilon 0.33

# r0:0.17000 r1:0.50000 r2:0.83000 | w0:0.33333 w1:0.33333 w2:0.33333
python fake_train.py \
  --outdir exps/latency/sd3-512_k3_nfe20 \
  --num-steps 11 \
  --num-points 3 \
  --guidance-rate 4.5 \
  --schedule-type flowmatch \
  --backend sd3 \
  --resolution 512 \
  --backend-options '{"model_name_or_path":"stabilityai/stable-diffusion-3-medium-diffusers"}' \
  --r-base 0.5 \
  --r-epsilon 0.33

################################ sd3-1024 ################################
# r0:0.5 | w0:1.00000
python fake_train.py \
  --outdir exps/latency/sd3-1024_k1_nfe20 \
  --num-steps 11 \
  --num-points 1 \
  --guidance-rate 4.5 \
  --schedule-type flowmatch \
  --backend sd3 \
  --resolution 1024 \
  --backend-options '{"model_name_or_path":"stabilityai/stable-diffusion-3-medium-diffusers"}' \
  --r-base 0.5 \
  --r-epsilon 0.0001

# r0:0.33500 r1:0.66500 | w0:0.50000 w1:0.50000
python fake_train.py \
  --outdir exps/latency/sd3-1024_k2_nfe20 \
  --num-steps 11 \
  --num-points 2 \
  --guidance-rate 4.5 \
  --schedule-type flowmatch \
  --backend sd3 \
  --resolution 1024 \
  --backend-options '{"model_name_or_path":"stabilityai/stable-diffusion-3-medium-diffusers"}' \
  --r-base 0.5 \
  --r-epsilon 0.33

# r0:0.17000 r1:0.50000 r2:0.83000 | w0:0.33333 w1:0.33333 w2:0.33333
python fake_train.py \
  --outdir exps/latency/sd3-1024_k3_nfe20 \
  --num-steps 11 \
  --num-points 3 \
  --guidance-rate 4.5 \
  --schedule-type flowmatch \
  --backend sd3 \
  --resolution 1024 \
  --backend-options '{"model_name_or_path":"stabilityai/stable-diffusion-3-medium-diffusers"}' \
  --r-base 0.5 \
  --r-epsilon 0.33

##########################################################################################

python latency_test.py \
  --predictor exps/latency/sd3-512_k1_nfe20/network-snapshot-000005.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds 0-99 \
  --max-batch-size 4 \
  --outdir ./latency_runs/sd3-512_k1_nfe20_b4 \
  --sampler epd_parallel \
  --latency-json ./latency_runs/sd3-512_k1_nfe20_b4/latency.json

python latency_test.py \
  --predictor exps/latency/sd3-512_k2_nfe20/network-snapshot-000005.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds 0-99 \
  --max-batch-size 4 \
  --outdir ./latency_runs/sd3-512_k2_nfe20_b4 \
  --sampler epd_parallel \
  --latency-json ./latency_runs/sd3-512_k2_nfe20_b4/latency.json

python latency_test.py \
  --predictor exps/latency/sd3-512_k3_nfe20/network-snapshot-000005.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds 0-99 \
  --max-batch-size 4 \
  --outdir ./latency_runs/sd3-512_k3_nfe20 \
  --sampler epd_parallel \
  --latency-json ./latency_runs/sd3-512_k3_nfe20/latency.json

##########################################################################################

python latency_test.py \
  --predictor exps/latency/sd3-1024_k1_nfe20/network-snapshot-000005.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds 0-99 \
  --max-batch-size 4 \
  --outdir ./latency_runs/sd3-1024_k1_nfe20 \
  --sampler epd_parallel \
  --latency-json ./latency_runs/sd3-1024_k1_nfe20/latency.json

python latency_test.py \
  --predictor exps/latency/sd3-1024_k2_nfe20/network-snapshot-000005.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds 0-99 \
  --max-batch-size 4 \
  --outdir ./latency_runs/sd3-1024_k2_nfe20 \
  --sampler epd_parallel \
  --latency-json ./latency_runs/sd3-1024_k2_nfe20/latency.json

python latency_test.py \
  --predictor exps/latency/sd3-1024_k3_nfe20/network-snapshot-000005.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds 0-99 \
  --max-batch-size 4 \
  --outdir ./latency_runs/sd3-1024_k3_nfe20 \
  --sampler epd_parallel \
  --latency-json ./latency_runs/sd3-1024_k3_nfe20/latency.json

##########################################################################################

python latency_test.py \
  --predictor exps/latency/sd15_k1_nfe20/network-snapshot-000005.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds 0-99 \
  --max-batch-size 4 \
  --outdir ./latency_runs/sd15_k1_nfe20 \
  --sampler epd_parallel \
  --latency-json ./latency_runs/sd15_k1_nfe20/latency.json

python latency_test.py \
  --predictor exps/latency/sd15_k2_nfe20/network-snapshot-000005.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds 0-99 \
  --max-batch-size 4 \
  --outdir ./latency_runs/sd15_k2_nfe20 \
  --sampler epd_parallel \
  --latency-json ./latency_runs/sd15_k2_nfe20/latency.json

python latency_test.py \
  --predictor exps/latency/sd15_k3_nfe20/network-snapshot-000005.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds 0-99 \
  --max-batch-size 4 \
  --outdir ./latency_runs/sd15_k3_nfe20 \
  --sampler epd_parallel \
  --latency-json ./latency_runs/sd15_k3_nfe20/latency.json
