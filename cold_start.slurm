#!/bin/bash
#SBATCH --job-name="protein"
#SBATCH --output="protein_%j.out"
#SBATCH --error="protein_%j.err"
#SBATCH --partition=gpuH200x8
#SBATCH --mem=512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest
#SBATCH --account=becj-delta-gpu
#SBATCH --no-requeue
#SBATCH -t 06:00:00

conda activate edm
module load cuda
cd /work/nvme/betk/zli42/EPD

train_model() {
    torchrun --standalone --nproc_per_node=4 --master_port=11111 \
        train.py \
        --dataset_name="$1" \
        --batch="$2" \
        --total_kimg="$3" \
        $SOLVER_FLAGS \
        $SCHEDULE_FLAGS \
        $ADDITIONAL_FLAGS \
        $GUIDANCE_FLAGS
}

SOLVER_FLAGS="--sampler_stu=epd --sampler_tea=dpm --num_steps=10 --M=1 --afs=False --scale_dir=0.00 --scale_time=0.0 --seed=0 --lr 0.01"
SCHEDULE_FLAGS="--schedule_type=discrete --schedule_rho=1"
ADDITIONAL_FLAGS="--max_order=2 --predict_x0=False --lower_order_final=True"
GUIDANCE_FLAGS="--guidance_type=cfg --guidance_rate=7.5"
train_model "ms_coco" 16 1