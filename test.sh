python fake_train.py \
  --outdir exps/badbadbad \
  --num-steps 7 \
  --num-points 1 \
  --guidance-rate 4.5 \
  --schedule-type flowmatch \
  --backend sd3 \
  --resolution 1024 \
  --backend-options '{"model_name_or_path":"stabilityai/stable-diffusion-3-medium-diffusers"}' \
  --r-base 0.5 \
  --r-epsilon 0.001

run_sd3_set() {
  local prompt="$1"
  local seeds="${2:-0}"

  python sample_sd3.py --predictor exps/fake-sd3-9-512/network-snapshot-000005.pkl \
    --seeds "$seeds" \
    --outdir training_step_images/1 \
    --prompt "$prompt"

  python sample_sd3.py --predictor exps/fake-sd3-11-512/network-snapshot-000005.pkl \
    --seeds "$seeds" \
    --outdir training_step_images/2 \
    --prompt "$prompt"

  python sample_sd3.py --predictor exps/20251210-005434-sd3_512/export/network-snapshot-export-step001000.pkl \
    --seeds "$seeds" \
    --outdir training_step_images/3 \
    --prompt "$prompt"

  python sample_sd3.py --predictor exps/20251210-005434-sd3_512/export/network-snapshot-export-step005000.pkl \
    --seeds "$seeds" \
    --outdir training_step_images/4 \
    --prompt "$prompt"

  python sample_sd3.py --predictor exps/20251210-005434-sd3_512/export/network-snapshot-export-step009000.pkl \
    --seeds "$seeds" \
    --outdir training_step_images/5 \
    --prompt "$prompt"
}


run_sd3_set 'A photo of a cow left of a stop sign.' '8'




python sample_sd3.py --predictor exps/20251210-011145-sd3_1024/export/network-snapshot-export-step007200.pkl \
  --seeds "1" \
  --outdir pipeline_images/GOOD \
  --prompt "A colorful blue tit bird perched on a detailed birch tree branch. The background is a bright, clear blue sky with fluffy white clouds, providing a high-contrast, well-lit environment. The sunlight is direct but soft, highlighting the vivid yellow and blue feathers of the bird and the texture of the white tree bark. The image feels open, crisp, and refreshing. 8k, high definition, nature photography."

python sample_sd3_baseline.py --sampler sd3 --resolution 1024 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt "A macro shot of a cute hamster standing on a stalk of wheat. The background is a bright, sun-drenched wheat field stretching into the distance, with visible individual stalks creating a textured golden pattern. The sun is bright, creating a halo effect around the animal's fur. The colors are warm, golden, and very bright. Sharp details on the whiskers and paws. Cinematic lighting, bright atmosphere." \
  --seeds "1" \
  --num-steps 6 \
  --outdir pipeline_images/BAD1

python sample_sd3_baseline.py --sampler sd3 --resolution 1024 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt "A macro shot of a cute hamster standing on a stalk of wheat. The background is a bright, sun-drenched wheat field stretching into the distance, with visible individual stalks creating a textured golden pattern. The sun is bright, creating a halo effect around the animal's fur. The colors are warm, golden, and very bright. Sharp details on the whiskers and paws. Cinematic lighting, bright atmosphere." \
  --seeds "1" \
  --num-steps 7 \
  --outdir pipeline_images/BAD2

python sample_sd3_baseline.py --sampler sd3 --resolution 1024 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt "A macro shot of a cute hamster standing on a stalk of wheat. The background is a bright, sun-drenched wheat field stretching into the distance, with visible individual stalks creating a textured golden pattern. The sun is bright, creating a halo effect around the animal's fur. The colors are warm, golden, and very bright. Sharp details on the whiskers and paws. Cinematic lighting, bright atmosphere." \
  --seeds "1" \
  --num-steps 8 \
  --outdir pipeline_images/BAD3

python sample_sd3_baseline.py --sampler sd3 --resolution 1024 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt "A macro shot of a cute hamster standing on a stalk of wheat. The background is a bright, sun-drenched wheat field stretching into the distance, with visible individual stalks creating a textured golden pattern. The sun is bright, creating a halo effect around the animal's fur. The colors are warm, golden, and very bright. Sharp details on the whiskers and paws. Cinematic lighting, bright atmosphere." \
  --seeds "1" \
  --num-steps 9 \
  --outdir pipeline_images/BAD4

python sample_sd3_baseline.py --sampler sd3 --resolution 1024 \
  --model-id "stabilityai/stable-diffusion-3-medium-diffusers" \
  --prompt "A macro shot of a cute hamster standing on a stalk of wheat. The background is a bright, sun-drenched wheat field stretching into the distance, with visible individual stalks creating a textured golden pattern. The sun is bright, creating a halo effect around the animal's fur. The colors are warm, golden, and very bright. Sharp details on the whiskers and paws. Cinematic lighting, bright atmosphere." \
  --seeds "1" \
  --num-steps 12 \
  --outdir pipeline_images/BAD5