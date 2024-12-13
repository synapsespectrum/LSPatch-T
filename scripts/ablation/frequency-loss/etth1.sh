#!/bin/bash

DATASET_PATH="dataset/ETT/"
SCRIPT_PATH="experiment.py"
CHECKPOINTS_PATH="./experiments/model_saved/checkpoints/"

# Check if the script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: $SCRIPT_PATH not found!"
    exit 1
fi

# Function to run experiment
run_experiment() {
    local model=$1
    local window_look_back=${2:-96}
    local is_pretrain=${3:-0}
    local is_finetune=${4:-0}
    local batch_size=${5:-32}
    local mask_ratio=${6:-0}
    local horizon=${7:-168}
    local patch_len=${8:-12}
    local stride=${9:-12}

    echo "$(date): Starting to train $model for forecasting horizon of $horizon "
    python -u $SCRIPT_PATH \
        --is_training 1 \
        --is_pretrain $is_pretrain \
        --is_finetune $is_finetune \
        --batch_size $batch_size \
        --root_path $DATASET_PATH \
        --data_path ETTh1.csv \
        --checkpoints $CHECKPOINTS_PATH \
        --model_id Loss \
        --model $model \
        --data ETTh1 \
        --freq h \
        --features M \
        --mask_ratio $mask_ratio \
        --patch_len $patch_len \
        --stride $stride \
        --seq_len $window_look_back \
        --label_len 48 \
        --pred_len $horizon \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des Exp \
        --itr 2

    if [ $? -eq 0 ]; then
        echo "$(date): Successfully trained $model for forecasting horizon of $horizon"
    else
        echo "$(date): Error occurred while training  $model for forecasting horizon of $horizon"
    fi
    echo "=================================================="
}

# Run experiments for LSPatchT
MODEL_NAME="LSPatchT"
for horizon in 24 48 96 168 192 336 720; do
    run_experiment $MODEL_NAME 96 1 1 64 0.4 $horizon 12 12
done