# epd （distillation / RL）
# python sample.py \
#     --predictor_path exps/20251030-235041-sd15_rl_base/export/network-snapshot-export-step005000.pkl \
#     --prompt-file src/prompts/test.txt \
#     --seeds "0-99" \
#     --batch 16 \
#     --outdir ./samples/test_rl_5000

#############################################################################

# python sample_baseline.py --sampler ddim \
#     --dataset-name ms_coco \
#     --prompt-file src/prompts/test.txt \
#     --seeds "0-999" --batch 16 \
#     --num-steps 19 --schedule-type time_uniform --schedule-rho 1.0 \
#     --outdir ./samples/test_ddim_nfe18

# python sample_baseline.py --sampler edm \
#     --dataset-name ms_coco \
#     --prompt-file src/prompts/test.txt \
#     --seeds "0-999" --batch 16 \
#     --num-steps 10 --schedule-type polynomial --schedule-rho 7.0 \
#     --outdir ./samples/test_edm_nfe18

# python sample_baseline.py --sampler dpm2 \
#     --dataset-name ms_coco \
#     --prompt-file src/prompts/test.txt \
#     --seeds "0-999" --batch 16 \
#     --num-steps 10 --schedule-type logsnr --schedule-rho 1.0 \
#     --outdir ./samples/test_dpm2_nfe18

# python sample_baseline.py --sampler ipndm \
#     --dataset-name ms_coco \
#     --prompt-file src/prompts/test.txt \
#     --seeds "0-999" --batch 16 \
#     --num-steps 19 --schedule-type time_uniform --schedule-rho 1.0 \
#     --max-order 3 \
#     --outdir ./samples/test_ipndm_nfe18

# python sample_baseline.py --sampler ddim \
#     --dataset-name ms_coco \
#     --prompt-file src/prompts/test.txt \
#     --seeds "0-999" --batch 16 \
#     --num-steps 51 --schedule-type time_uniform --schedule-rho 1.0 \
#     --outdir ./samples/test_ddim_nfe50

# python sample_baseline.py --sampler edm \
#     --dataset-name ms_coco \
#     --prompt-file src/prompts/test.txt \
#     --seeds "0-999" --batch 16 \
#     --num-steps 26 --schedule-type polynomial --schedule-rho 7.0 \
#     --outdir ./samples/test_edm_nfe50

# python sample_baseline.py --sampler dpm2 \
#     --dataset-name ms_coco \
#     --prompt-file src/prompts/test.txt \
#     --seeds "0-999" --batch 16 \
#     --num-steps 26 --schedule-type logsnr --schedule-rho 1.0 \
#     --outdir ./samples/test_dpm2_nfe50

# python sample_baseline.py --sampler ipndm \
#     --dataset-name ms_coco \
#     --prompt-file src/prompts/test.txt \
#     --seeds "0-999" --batch 16 \
#     --num-steps 51 --schedule-type time_uniform --schedule-rho 1.0 \
#     --max-order 3 \
#     --outdir ./samples/test_ipndm_nfe50

python sample_baseline.py --sampler ddim \
    --dataset-name ms_coco \
    --prompt-file src/prompts/test.txt \
    --seeds "0-999" --batch 16 \
    --num-steps 19 --schedule-type discrete --schedule-rho 1.0 \
    --outdir ./samples/test_ddim_nfe18_discrete

python sample_baseline.py --sampler dpm2 \
    --dataset-name ms_coco \
    --prompt-file src/prompts/test.txt \
    --seeds "0-999" --batch 16 \
    --num-steps 10 --schedule-type discrete --schedule-rho 1.0 \
    --outdir ./samples/test_dpm2_nfe18_discrete

python sample_baseline.py --sampler ipndm \
    --dataset-name ms_coco \
    --prompt-file src/prompts/test.txt \
    --seeds "0-999" --batch 16 \
    --num-steps 19 --schedule-type discrete --schedule-rho 1.0 \
    --max-order 3 \
    --outdir ./samples/test_ipndm_nfe18_discrete

python sample_baseline.py --sampler ddim \
    --dataset-name ms_coco \
    --prompt-file src/prompts/test.txt \
    --seeds "0-999" --batch 16 \
    --num-steps 51 --schedule-type discrete --schedule-rho 1.0 \
    --outdir ./samples/test_ddim_nfe50_discrete

python sample_baseline.py --sampler dpm2 \
    --dataset-name ms_coco \
    --prompt-file src/prompts/test.txt \
    --seeds "0-999" --batch 16 \
    --num-steps 26 --schedule-type discrete --schedule-rho 1.0 \
    --outdir ./samples/test_dpm2_nfe50_discrete

python sample_baseline.py --sampler ipndm \
    --dataset-name ms_coco \
    --prompt-file src/prompts/test.txt \
    --seeds "0-999" --batch 16 \
    --num-steps 51 --schedule-type discrete --schedule-rho 1.0 \
    --max-order 3 \
    --outdir ./samples/test_ipndm_nfe50_discrete

python sample.py \
    --predictor_path exps/00036-ms_coco-10-36-epd-dpm-1-discrete/network-snapshot-000005.pkl \
    --prompt-file src/prompts/test.txt \
    --seeds "0-999" \
    --batch 16 \
    --outdir ./samples/test_rl_0_nfe18

python sample.py \
    --predictor_path exps/20251030-235041-sd15_rl_base/export/network-snapshot-export-step001700.pkl \
    --prompt-file src/prompts/test.txt \
    --seeds "0-999" \
    --batch 16 \
    --outdir ./samples/test_rl_1700_nfe18

python sample.py \
    --predictor_path exps/20251030-235041-sd15_rl_base/export/network-snapshot-export-step002800.pkl \
    --prompt-file src/prompts/test.txt \
    --seeds "0-999" \
    --batch 16 \
    --outdir ./samples/test_rl_2800_nfe18

python sample.py \
    --predictor_path exps/20251030-235041-sd15_rl_base/export/network-snapshot-export-step005000.pkl \
    --prompt-file src/prompts/test.txt \
    --seeds "0-999" \
    --batch 16 \
    --outdir ./samples/test_rl_5000_nfe18

python sample.py \
    --predictor_path exps/20251030-235041-sd15_rl_base/export/network-snapshot-export-step005450.pkl \
    --prompt-file src/prompts/test.txt \
    --seeds "0-999" \
    --batch 16 \
    --outdir ./samples/test_rl_5450_nfe18

#############################################################################

# python -m training.ppo.scripts.score_hps \
#     --images samples/test_ddim_nfe18 \
#     --pattern "**/*.png" \
#     --prompts src/prompts/test.txt \
#     --weights weights/HPS_v2.1_compressed.pt \
#     --output-json results/test_ddim_nfe18.json

# python -m training.ppo.scripts.score_hps \
#     --images samples/test_edm_nfe18 \
#     --pattern "**/*.png" \
#     --prompts src/prompts/test.txt \
#     --weights weights/HPS_v2.1_compressed.pt \
#     --output-json results/test_edm_nfe18.json

# python -m training.ppo.scripts.score_hps \
#     --images samples/test_dpm2_nfe18 \
#     --pattern "**/*.png" \
#     --prompts src/prompts/test.txt \
#     --weights weights/HPS_v2.1_compressed.pt \
#     --output-json results/test_dpm2_nfe18.json

# python -m training.ppo.scripts.score_hps \
#     --images samples/test_ipndm_nfe18 \
#     --pattern "**/*.png" \
#     --prompts src/prompts/test.txt \
#     --weights weights/HPS_v2.1_compressed.pt \
#     --output-json results/test_ipndm_nfe18.json

# python -m training.ppo.scripts.score_hps \
#     --images samples/test_ddim_nfe50 \
#     --pattern "**/*.png" \
#     --prompts src/prompts/test.txt \
#     --weights weights/HPS_v2.1_compressed.pt \
#     --output-json results/test_ddim_nfe50.json

# python -m training.ppo.scripts.score_hps \
#     --images samples/test_edm_nfe50 \
#     --pattern "**/*.png" \
#     --prompts src/prompts/test.txt \
#     --weights weights/HPS_v2.1_compressed.pt \
#     --output-json results/test_edm_nfe50.json

# python -m training.ppo.scripts.score_hps \
#     --images samples/test_dpm2_nfe50 \
#     --pattern "**/*.png" \
#     --prompts src/prompts/test.txt \
#     --weights weights/HPS_v2.1_compressed.pt \
#     --output-json results/test_dpm2_nfe50.json

# python -m training.ppo.scripts.score_hps \
#     --images samples/test_ipndm_nfe50 \
#     --pattern "**/*.png" \
#     --prompts src/prompts/test.txt \
#     --weights weights/HPS_v2.1_compressed.pt \
#     --output-json results/test_ipndm_nfe50.json

python -m training.ppo.scripts.score_hps \
    --images samples/test_ddim_nfe18_discrete \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/HPS_v2.1_compressed.pt \
    --output-json results/test_ddim_nfe18_discrete.json

python -m training.ppo.scripts.score_hps \
    --images samples/test_dpm2_nfe18_discrete \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/HPS_v2.1_compressed.pt \
    --output-json results/test_dpm2_nfe18_discrete.json

python -m training.ppo.scripts.score_hps \
    --images samples/test_ipndm_nfe18_discrete \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/HPS_v2.1_compressed.pt \
    --output-json results/test_ipndm_nfe18_discrete.json

python -m training.ppo.scripts.score_hps \
    --images samples/test_ddim_nfe50_discrete \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/HPS_v2.1_compressed.pt \
    --output-json results/test_ddim_nfe50_discrete.json

python -m training.ppo.scripts.score_hps \
    --images samples/test_dpm2_nfe50_discrete \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/HPS_v2.1_compressed.pt \
    --output-json results/test_dpm2_nfe50_discrete.json

python -m training.ppo.scripts.score_hps \
    --images samples/test_ipndm_nfe50_discrete \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/HPS_v2.1_compressed.pt \
    --output-json results/test_ipndm_nfe50_discrete.json

python -m training.ppo.scripts.score_hps \
    --images samples/test_rl_0_nfe18 \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/HPS_v2.1_compressed.pt \
    --output-json results/test_rl_0_nfe18.json

python -m training.ppo.scripts.score_hps \
    --images samples/test_rl_1700_nfe18 \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/HPS_v2.1_compressed.pt \
    --output-json results/test_rl_1700_nfe18.json

python -m training.ppo.scripts.score_hps \
    --images samples/test_rl_2800_nfe18 \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/HPS_v2.1_compressed.pt \
    --output-json results/test_rl_2800_nfe18.json

python -m training.ppo.scripts.score_hps \
    --images samples/test_rl_5000_nfe18 \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/HPS_v2.1_compressed.pt \
    --output-json results/test_rl_5000_nfe18.json

python -m training.ppo.scripts.score_hps \
    --images samples/test_rl_5450_nfe18 \
    --pattern "**/*.png" \
    --prompts src/prompts/test.txt \
    --weights weights/HPS_v2.1_compressed.pt \
    --output-json results/test_rl_5450_nfe18.json