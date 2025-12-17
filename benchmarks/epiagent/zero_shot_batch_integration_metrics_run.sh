#!/bin/bash

# python zero_shot_batch_integration_metrics.py \
#     --dataset_name Li2023b \
#     --batch_key "Batch (HSC)" \
#     --root /scratch/naimer/github/project-2-team-1/EpiAgent/data \
#     --seed 42 \
#     --n_cores 128

python zero_shot_batch_integration_metrics.py \
    --dataset_name Kanemaru2023 \
    --batch_key "Batch (HSC)" \
    --root /scratch/naimer/github/project-2-team-1/EpiAgent/data \
    --seed 42 \
    --n_cores 128
