#!/bin/bash

# Parallel Cell Embedding Script
# This script runs 4 separate processes on 4 available GPUs

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /home/naimer/miniforge3/envs/chromfound

# Configuration
DATA_PATH="sample_data/PBMC169K/atac_pbmc_benchmark_VIB_10xv1_1.h5ad"
OUTPUT_PATH="sample_data/PBMC169K/cell_embedding"
PRETRAIN_CHECKPOINT_PATH="src/checkpoints"
PRETRAIN_MODEL_FILE="model.pt"
PRETRAIN_CONFIG_FILE="chromfd_pretrain.yaml"
BATCH_SIZE=4
CELL_TYPE_COL="celltype"
NUM_SPLITS=1

# Available GPUs (not currently in use)
GPUS=(5)

echo "Starting parallel cell embedding across ${NUM_SPLITS} GPUs: ${GPUS[@]}"
echo "========================================================================"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"

# Launch processes in parallel
for i in {0..0}; do
    GPU=${GPUS[$i]}
    echo "Launching process $((i+1))/$NUM_SPLITS on GPU $GPU..."
    
    ~/miniforge3/envs/chromfound/bin/python -m src.cell_embedding \
        --data_path "$DATA_PATH" \
        --output_path "$OUTPUT_PATH" \
        --pretrain_checkpoint_path "$PRETRAIN_CHECKPOINT_PATH" \
        --pretrain_model_file "$PRETRAIN_MODEL_FILE" \
        --pretrain_config_file "$PRETRAIN_CONFIG_FILE" \
        --batch_size $BATCH_SIZE \
        --cell_type_col "$CELL_TYPE_COL" \
        --local_rank $GPU \
        --num_splits $NUM_SPLITS \
        --split_idx $i \
        > "${OUTPUT_PATH}/log_split_${i}.txt" 2>&1 &
    
    # Store process ID
    PIDS[$i]=$!
done

echo "All processes launched. PIDs: ${PIDS[@]}"
echo "Waiting for all processes to complete..."

# Wait for all processes to finish
for i in {0..3}; do
    wait ${PIDS[$i]}
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Process $((i+1))/$NUM_SPLITS (PID ${PIDS[$i]}) completed successfully"
    else
        echo "Process $((i+1))/$NUM_SPLITS (PID ${PIDS[$i]}) failed with exit code $EXIT_CODE"
    fi
done

echo "========================================================================"
echo "All processes completed. Check logs in ${OUTPUT_PATH}/log_split_*.txt"
echo "Run merge script to combine results: python merge_embeddings.py"

