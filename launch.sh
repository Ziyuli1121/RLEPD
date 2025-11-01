#!/bin/bash

###############################
### A. Train EPD Predictor  ###
###############################

# Hardware Note: 
# - Use 4xA800 for LSUN_Bedroom_ldm & Stable Diffusion
# - Use 4xRTX4090 for other experiments
# - Adjust batch size according to your GPU memory

# Note on num_steps:
# - Original steps = num_steps (N)
# - Final steps = 2*(N-1) (EPD inserts intermediate steps)
# - NFE = 5 (afs=True) or 6 (afs=False)

train_model() {
    torchrun --standalone --nproc_per_node=8 --master_port=11111 \
        train.py \
        --dataset_name="$1" \
        --batch="$2" \
        --total_kimg="$3" \
        $SOLVER_FLAGS \
        $SCHEDULE_FLAGS \
        $ADDITIONAL_FLAGS \
        $GUIDANCE_FLAGS
}

## A.1 CIFAR-10 ##
SOLVER_FLAGS="--sampler_stu=epd --sampler_tea=dpm --num_steps=4 --M=3 --afs=True --scale_dir=0.05 --scale_time=0.05 --seed=0 --lr 0.2 --coslr"
SCHEDULE_FLAGS="--schedule_type=time_uniform --schedule_rho=1"
train_model "cifar10" 128 10

SOLVER_FLAGS="--sampler_stu=ipndm --sampler_tea=ipndm --num_steps=4 --M=3 --afs=True --scale_dir=0.05 --scale_time=0.2 --seed=0 --lr 0.2 --coslr"
SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
ADDITIONAL_FLAGS="--max_order=4"
train_model "cifar10" 128 10

## A.2 FFHQ ##
SOLVER_FLAGS="--sampler_stu=epd --sampler_tea=dpm --num_steps=4 --M=3 --afs=True --scale_dir=0.05 --scale_time=0.05 --seed=0 --lr 0.2 --coslr"
SCHEDULE_FLAGS="--schedule_type=time_uniform --schedule_rho=1"
train_model "ffhq" 64 10

## A.3 ImageNet-64 ##
SOLVER_FLAGS="--sampler_stu=epd --sampler_tea=dpm --num_steps=4 --M=1 --afs=True --scale_dir=0.05 --scale_time=0.05 --seed=0 --lr 0.2 --coslr"
SCHEDULE_FLAGS="--schedule_type=time_uniform --schedule_rho=1"
train_model "imagenet64" 64 10

## A.4 LSUN Bedroom (LDM) ##
SOLVER_FLAGS="--sampler_stu=epd --sampler_tea=dpm --num_steps=4 --M=3 --afs=True --scale_dir=0.1 --scale_time=0 --seed=0 --lr 0.02 --coslr"
SCHEDULE_FLAGS="--schedule_type=discrete --schedule_rho=1"
ADDITIONAL_FLAGS="--max_order=3 --predict_x0=False --lower_order_final=True"
GUIDANCE_FLAGS="--guidance_type=uncond --guidance_rate=1"
train_model "lsun_bedroom_ldm" 64 10

## A.5 Stable Diffusion ##
# Note: NFE doubles due to classifier-free guidance
SOLVER_FLAGS="--sampler_stu=epd --sampler_tea=dpm --num_steps=4 --M=3 --afs=True --scale_dir=0.05 --scale_time=0.2 --seed=0 --lr 0.01"
SCHEDULE_FLAGS="--schedule_type=discrete --schedule_rho=1"
ADDITIONAL_FLAGS="--max_order=2 --predict_x0=False --lower_order_final=True"
GUIDANCE_FLAGS="--guidance_type=cfg --guidance_rate=7.5"
train_model "ms_coco" 32 5

SOLVER_FLAGS="--sampler_stu=epd --sampler_tea=dpm --num_steps=10 --M=1 --afs=False --scale_dir=0.00 --scale_time=0.0 --seed=0 --lr 0.01"
SCHEDULE_FLAGS="--schedule_type=discrete --schedule_rho=1"
ADDITIONAL_FLAGS="--max_order=2 --predict_x0=False --lower_order_final=True"
GUIDANCE_FLAGS="--guidance_type=cfg --guidance_rate=7.5"
train_model "ms_coco" 32 5

SOLVER_FLAGS="--sampler_stu=epd --sampler_tea=dpm --num_steps=10 --M=3 --afs=False --scale_dir=0.00 --scale_time=0.0 --seed=0 --lr 0.01"
SCHEDULE_FLAGS="--schedule_type=discrete --schedule_rho=1"
ADDITIONAL_FLAGS="--max_order=2 --predict_x0=False --lower_order_final=True"
GUIDANCE_FLAGS="--guidance_type=cfg --guidance_rate=7.5"
train_model "ms_coco" 32 5


#################################
### B. Generate Samples for FID ###
#################################

# Trained predictors are saved in ./exps/ (5-digit experiment numbers)
# Use either:
# --predictor_path=/full/path
# --predictor_path=EXP_NUMBER (e.g., 00000)

generate_samples() {
    torchrun --standalone --nproc_per_node=1 --master_port=22222 \
        sample.py \
        --predictor_path="$1" \
        --batch="$2" \
        --seeds="$3"
}

## B.1 Standard Datasets ##
generate_samples 0 128 "0-49999"

## B.2 Stable Diffusion ##
generate_samples 0 32 "0-29"
generate_samples "/work/nvme/betk/zli42/RLEPD/exps/00036-ms_coco-10-36-epd-dpm-1-discrete/network-snapshot-000005.pkl" 2 "0-29"


##########################
### C. Evaluation ###
##########################

## C.1 FID Calculation ##
python fid.py calc \
    --images="/work/nvme/betk/zli42/EPD/samples/ms_coco/epd_parallel_nfe36_npoints_2" \
    --ref="/work/nvme/betk/zli42/EPD/ms_coco-512x512.npz"

