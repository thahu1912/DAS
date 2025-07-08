#!/bin/bash

DATA_ROOT='data'


# ===================> Tri-Hyp [Distance Mining] <===================
echo "CUB200 - Tri-Hyp with Distance Mining"
CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
    --gpu 0 1 \
    --dataset cub200 \
    --kernels 6 \
    --source $DATA_ROOT \
    --n_epochs 300 \
    --tau 200 250 \
    --seed 0 \
    --bs 112 \
    --samples_per_class 2 \
    --loss tri_hyp \
    --batch_mining distance \
    --arch resnet50_frozen_tri_hyp_cub200_distance_300epoch_200_250 \
    --hyperbolic_c 0.5 \
    --k_neighbors 8 \
    --temperature 0.05 \
    --label_weight 0.7 \
    --train_c

# ===================> Tri-Hyp [Semi-Hard Mining] <===================
echo "CUB200 - Tri-Hyp with Semi-Hard Mining"
CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
    --gpu 0 1 \
    --dataset cub200 \
    --kernels 6 \
    --source $DATA_ROOT \
    --n_epochs 300 \
    --tau 200 250 \
    --seed 0 \
    --bs 112 \
    --samples_per_class 2 \
    --loss tri_hyp \
    --batch_mining semihard \
    --arch resnet50_frozen_tri_hyp_cub200_semihard_300epoch_200_250 \
    --hyperbolic_c 0.5 \
    --k_neighbors 8 \
    --temperature 0.05 \
    --label_weight 0.7 \
    --train_c



echo "CARS196 - Tri-Hyp with Distance Mining"
CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
    --gpu 0 1 \
    --dataset cars196 \
    --kernels 6 \
    --source $DATA_ROOT \
    --n_epochs 300 \
    --tau 200 250 \
    --seed 0 \
    --bs 112 \
    --samples_per_class 2 \
    --loss tri_hyp \
    --batch_mining distance \
    --arch resnet50_frozen_tri_hyp_cars196_distance_300epoch_200_250 \
    --hyperbolic_c 0.5 \
    --k_neighbors 8 \
    --temperature 0.05 \
    --label_weight 0.7 \
    --train_c

# ===================> Tri-Hyp [Semi-Hard Mining] <===================
echo "CARS196 - Tri-Hyp with Semi-Hard Mining"
CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
    --gpu 0 1 \
    --dataset cars196 \
    --kernels 6 \
    --source $DATA_ROOT \
    --n_epochs 300 \
    --tau 200 250 \
    --seed 0 \
    --bs 112 \
    --samples_per_class 2 \
    --loss tri_hyp \
    --batch_mining semihard \
    --arch resnet50_frozen_tri_hyp_cars196_semihard_300epoch_200_250 \
    --hyperbolic_c 0.5 \
    --k_neighbors 8 \
    --temperature 0.05 \
    --label_weight 0.7 \
    --train_c



#################################### SOP ####################################
echo "Starting SOP training..."

# ===================> Tri-Hyp [Distance Mining] <===================
echo "SOP - Tri-Hyp with Distance Mining"
CUDA_VISIBLE_DEVICES=0,1 python3 -u main.py \
    --gpu 0 1 \
    --dataset online_products \
    --kernels 6 \
    --source $DATA_ROOT \
    --n_epochs 300 \
    --tau 200 250 \
    --seed 0 \
    --bs 112 \
    --samples_per_class 2 \
    --loss tri_hyp \
    --batch_mining distance \
    --arch resnet50_frozen_tri_hyp_sop_distance_300epoch_200_250 \
    --hyperbolic_c 0.5 \
    --k_neighbors 8 \
    --temperature 0.05 \
    --label_weight 0.7 \
    --train_c

# ===================> Tri-Hyp [Semi-Hard Mining] <===================
echo "SOP - Tri-Hyp with Semi-Hard Mining"
CUDA_VISIBLE_DEVICES=0,1 python3 -u main.py \
    --gpu 0 1 \
    --dataset online_products \
    --kernels 6 \
    --source $DATA_ROOT \
    --n_epochs 300 \
    --tau 200 250 \
    --seed 0 \
    --bs 112 \
    --samples_per_class 2 \
    --loss tri_hyp \
    --batch_mining semihard \
    --arch resnet50_frozen_tri_hyp_sop_semihard_300epoch_200_250 \
    --hyperbolic_c 0.5 \
    --k_neighbors 8 \
    --temperature 0.05 \
    --label_weight 0.7 \
    --train_c



echo "=========================================="
echo "All Tri-Hyp training completed!"
echo "==========================================" 