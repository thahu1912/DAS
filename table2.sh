DATA_ROOT='data'

#################################### CUB ####################################
# ===================> Triplet [S] <===================
# baseline
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --gpu 0 1  --dataset cub200 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss triplet --batch_mining semihard --arch resnet50_frozen_triplet_semihard_normalize_300epoch_200_250
# baseline + DAS
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --gpu 0 1  --dataset cub200 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss triplet --batch_mining semihard --arch resnet50_frozen_triplet_semihard_normalize_300epoch_200_250_embeddingProducer_DAS --embedding_producer_used --produce_type DAS --das_dfs_scale_left 0.99 --das_dfs_scale_right 1.01 --detach

# ===================> Triplet [D] <===================
# baseline
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --gpu 0 1  --dataset cub200 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss triplet --batch_mining distance --arch resnet50_frozen_normalize_triplet_distance_300epoch_200_250
# baseline + DAS
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --gpu 0 1  --dataset cub200 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss triplet --batch_mining distance --arch resnet50_frozen_normalize_triplet_distance_300epoch_200_250_embeddingProducer_DAS --embedding_producer_used --produce_type DAS --das_dfs_scale_left 0.99 --das_dfs_scale_right 1.01 --detach

# ===================> Contrastive [D] <===================
# baseline
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --gpu 0 1  --dataset cub200 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss contrastive --batch_mining distance --arch resnet50_frozen_normalize_contrastive_distance_300epoch_200_250
# baseline + DAS
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --gpu 0 1  --dataset cub200 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss contrastive --batch_mining distance --arch resnet50_frozen_normalize_contrastive_distance_300epoch_200_250_embeddingProducer_DAS --embedding_producer_used --produce_type DAS --das_dfs_scale_left 0.99 --das_dfs_scale_right 1.01 --detach

# ===================> Margin <===================
# baseline
CUDA_VISIBLE_DEVICES=0,1 python3 -u main.py --gpu 0 1  --dataset cub200 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_frozen_normalize_margin_distance_300epoch_200_250
# baseline + DAS
CUDA_VISIBLE_DEVICES=0,1 python3 -u main.py --gpu 0 1  --dataset cub200 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_frozen_normalize_margin_distance_300epoch_200_250_embeddingProducer_DAS --embedding_producer_used --produce_type DAS --das_dfs_scale_left 0.99 --das_dfs_scale_right 1.01 --detach

# ===================> GenLifted <===================
# baseline
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --gpu 0 1  --dataset cub200 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss lifted --batch_mining lifted --arch resnet50_frozen_lifted_300epoch_200_250
# baseline + DAS
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --gpu 0 1  --dataset cub200 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss lifted --batch_mining lifted --arch resnet50_frozen_lifted_300epoch_200_250_embeddingProducer_DAS_notNorm --embedding_producer_used --produce_type DAS --das_dfs_scale_left 0.99 --das_dfs_scale_right 1.01 --detach --das_not_normalize

# ===================> N-Pair <===================
# baseline
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --gpu 0 1  --dataset cub200 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss npair --batch_mining npair --arch resnet50_frozen_npair_300epoch_200_250
# baseline + DAS
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --gpu 0 1  --dataset cub200 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss npair --batch_mining npair --arch resnet50_frozen_npair_300epoch_200_250_embeddingProducer_DAS_notNorm --embedding_producer_used --produce_type DAS --das_dfs_scale_left 0.99 --das_dfs_scale_right 1.01 --detach --das_not_normalize

# ===================> MS <===================
# baseline
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --gpu 0 1  --dataset cub200 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss multisimilarity --arch resnet50_frozen_normalize_multiSim_300epoch_200_250
# baseline + DAS
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --gpu 0 1  --dataset cub200 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss multisimilarity --arch resnet50_frozen_normalize_multiSim_300epoch_200_250_embeddingProducer_DAS --embedding_producer_used --produce_type DAS --das_dfs_scale_left 0.99 --das_dfs_scale_right 1.01 --detach

#################################### CARS ####################################
# ===================> Triplet [S] <===================
# baseline
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --gpu 0 1  --dataset cars196 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss triplet --batch_mining semihard --arch resnet50_frozen_triplet_semihard_normalize_300epoch_200_250
# baseline + DAS
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --gpu 0 1  --dataset cars196 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss triplet --batch_mining semihard --arch resnet50_frozen_triplet_semihard_normalize_300epoch_200_250_embeddingProducer_DAS --embedding_producer_used --produce_type DAS --das_dfs_scale_left 0.99 --das_dfs_scale_right 1.01 --detach

# ===================> Triplet [D] <===================
# baseline
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --gpu 0 1  --dataset cars196 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss triplet --batch_mining distance --arch resnet50_frozen_normalize_triplet_distance_300epoch_200_250
# baseline + DAS
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --gpu 0 1  --dataset cars196 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss triplet --batch_mining distance --arch resnet50_frozen_normalize_triplet_distance_300epoch_200_250_embeddingProducer_DAS --embedding_producer_used --produce_type DAS --das_dfs_scale_left 0.99 --das_dfs_scale_right 1.01 --detach

# ===================> Contrastive [D] <===================
# baseline
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --gpu 0 1  --dataset cars196 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss contrastive --batch_mining distance --arch resnet50_frozen_normalize_contrastive_distance_300epoch_200_250
# baseline + DAS
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --gpu 0 1  --dataset cars196 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss contrastive --batch_mining distance --arch resnet50_frozen_normalize_contrastive_distance_300epoch_200_250_embeddingProducer_DAS --embedding_producer_used --produce_type DAS --das_dfs_scale_left 0.99 --das_dfs_scale_right 1.01 --detach

# ===================> Margin <===================
# baseline
CUDA_VISIBLE_DEVICES=0,1 python3 -u main.py --gpu 0 1  --dataset cars196 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_frozen_normalize_margin_distance_300epoch_200_250
# baseline + DAS
CUDA_VISIBLE_DEVICES=0,1 python3 -u main.py --gpu 0 1  --dataset cars196 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_frozen_normalize_margin_distance_300epoch_200_250_embeddingProducer_DAS --embedding_producer_used --produce_type DAS --das_dfs_scale_left 0.99 --das_dfs_scale_right 1.01 --detach

# ===================> GenLifted <===================
# baseline
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --gpu 0 1  --dataset cars196 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss lifted --batch_mining lifted --arch resnet50_frozen_lifted_300epoch_200_250
# baseline + DAS
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --gpu 0 1  --dataset cars196 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss lifted --batch_mining lifted --arch resnet50_frozen_lifted_300epoch_200_250_embeddingProducer_DAS_notNorm --embedding_producer_used --produce_type DAS --das_dfs_scale_left 0.99 --das_dfs_scale_right 1.01 --detach --das_not_normalize

# ===================> N-Pair <===================
# baseline
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --gpu 0 1  --dataset cars196 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss npair --batch_mining npair --arch resnet50_frozen_npair_300epoch_200_250
# baseline + DAS
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --gpu 0 1  --dataset cars196 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss npair --batch_mining npair --arch resnet50_frozen_npair_300epoch_200_250_embeddingProducer_DAS_notNorm --embedding_producer_used --produce_type DAS --das_dfs_scale_left 0.99 --das_dfs_scale_right 1.01 --detach --das_not_normalize

# ===================> MS <===================
# baseline
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --gpu 0 1  --dataset cars196 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss multisimilarity --arch resnet50_frozen_normalize_multiSim_300epoch_200_250
# baseline + DAS
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --gpu 0 1  --dataset cars196 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss multisimilarity --arch resnet50_frozen_normalize_multiSim_300epoch_200_250_embeddingProducer_DAS --embedding_producer_used --produce_type DAS --das_dfs_scale_left 0.99 --das_dfs_scale_right 1.01 --detach

#################################### SOP ####################################
# ===================> Triplet [S] <===================
# baseline
CUDA_VISIBLE_DEVICES=0,1 python3 -u main.py --gpu 0 1  --dataset online_products --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss triplet --batch_mining semihard --arch resnet50_frozen_normalize_triplet_semihard_300epoch_200_250
# baseline + DAS
CUDA_VISIBLE_DEVICES=0,1 python3 -u main.py --gpu 0 1  --dataset online_products --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss triplet --batch_mining semihard --arch resnet50_frozen_normalize_triplet_semihard_300epoch_200_250_embeddingProducer_DAS --embedding_producer_used --produce_type DAS --das_mts_scale 0.1 --aux_loss --detach

# ===================> Triplet [D] <===================
# baseline
CUDA_VISIBLE_DEVICES=0,1 python3 -u main.py --gpu 0 1  --dataset online_products --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss triplet --batch_mining distance --arch resnet50_frozen_normalize_triplet_distance_300epoch_200_250
# baseline + DAS
CUDA_VISIBLE_DEVICES=0,1 python3 -u main.py --gpu 0 1  --dataset online_products --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss triplet --batch_mining distance --arch resnet50_frozen_normalize_triplet_distance_300epoch_200_250_embeddingProducer_DAS --embedding_producer_used --produce_type DAS --das_mts_scale 0.1 --aux_loss --detach

# ===================> Contrastive [D] <===================
# baseline
CUDA_VISIBLE_DEVICES=0,1 python3 -u main.py --gpu 0 1  --dataset online_products --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss contrastive --batch_mining distance --arch resnet50_frozen_normalize_contrastive_distance_300epoch_200_250
# baseline + DAS
CUDA_VISIBLE_DEVICES=0,1 python3 -u main.py --gpu 0 1  --dataset online_products --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss contrastive --batch_mining distance --arch resnet50_frozen_normalize_contrastive_distance_300epoch_200_250_embeddingProducer_DAS --embedding_producer_used --produce_type DAS --das_mts_scale 0.1 

# ===================> Margin <===================
# baseline
CUDA_VISIBLE_DEVICES=0,1 python3 -u main.py --gpu 0 1  --dataset online_products --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_frozen_normalize_margin_distance_300epoch_200_250
# baseline + DAS
CUDA_VISIBLE_DEVICES=0,1 python3 -u main.py --gpu 0 1  --dataset online_products --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_frozen_normalize_margin_distance_300epoch_200_250_embeddingProducer_DAS --embedding_producer_used --produce_type DAS

# ===================> GenLifted <===================
# baseline
CUDA_VISIBLE_DEVICES=0,1 python3 -u main.py --gpu 0 1  --dataset online_products --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss lifted --batch_mining lifted --arch resnet50_frozen_lifted_300epoch_200_250
# baseline + DAS
CUDA_VISIBLE_DEVICES=0,1 python3 -u main.py --gpu 0 1  --dataset online_products --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss lifted --batch_mining lifted --arch resnet50_frozen_lifted_300epoch_200_250_embeddingProducer_DAS_notNorm --embedding_producer_used --produce_type DAS --das_dfs_scale_left 0.99 --das_dfs_scale_right 1.01 --das_mts_scale 0.1 --das_not_normalize

# ===================> N-Pair <===================
# baseline
CUDA_VISIBLE_DEVICES=0,1 python3 -u main.py --gpu 0 1  --dataset online_products --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss npair --batch_mining npair --arch resnet50_frozen_npair_300epoch_200_250
# baseline + DAS
CUDA_VISIBLE_DEVICES=0,1 python3 -u main.py --gpu 0 1  --dataset online_products --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss npair --batch_mining npair --arch resnet50_frozen_npair_300epoch_200_250_embeddingProducer_DAS_notNorm --embedding_producer_used --produce_type DAS --das_dfs_scale_left 0.99 --das_dfs_scale_right 1.01 --das_mts_scale 0.1 --das_not_normalize

# ===================> MS <===================
# baseline
CUDA_VISIBLE_DEVICES=0,1 python3 -u main.py --gpu 0 1  --dataset online_products --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss multisimilarity --arch resnet50_frozen_normalize_multiSim_300epoch_200_250
# baseline + DAS
CUDA_VISIBLE_DEVICES=0,1 python3 -u main.py --gpu 0 1  --dataset online_products --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --bs 112 --samples_per_class 2 --loss multisimilarity --arch resnet50_frozen_normalize_multiSim_300epoch_200_250_embeddingProducer_DAS --embedding_producer_used --produce_type DAS --das_mts_scale 0.1 