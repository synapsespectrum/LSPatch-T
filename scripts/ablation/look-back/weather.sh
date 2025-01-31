#!/bin/bash

DATASET_PATH="dataset/"
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
    local window_look_back=$2
    local is_pretrain=${3:-0}
    local is_finetune=${4:-0}
    local data_path=${5:-"weather.csv"}
    local data=${6:-"Weather"}
    local batch_size=${7:-32}
    local mask_ratio=${8:-0}
    local patch_len=${9:-16}
    local stride=${10:-16}

    echo "$(date): Starting to train $model with look back window of $window_look_back"
    python -u $SCRIPT_PATH \
        --is_training 1 \
        --is_pretrain $is_pretrain \
        --is_finetune $is_finetune \
        --batch_size $batch_size \
        --root_path $DATASET_PATH \
        --data_path $data_path \
        --checkpoints $CHECKPOINTS_PATH \
        --model_id Look \
        --model $model \
        --data $data \
        --freq h \
        --features M \
        --mask_ratio $mask_ratio \
        --patch_len $patch_len \
        --stride $stride \
        --seq_len $window_look_back \
        --label_len 48 \
        --pred_len 168 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des Exp \
        --itr 2

    if [ $? -eq 0 ]; then
        echo "$(date): Successfully finished training $model with look back window of $window_look_back"
    else
        echo "$(date): Error occurred while training $model with look back window of $window_look_back"
    fi
    echo "=================================================="
}

# Run experiments for iTransformer, PatchTST, Crossformer
for model in iTransformer PatchTST Crossformer; do
    for window_look_back in 48 96 192 336 720; do
        run_experiment $model $window_look_back
    done
done

# Run experiments for LSPatchT
MODEL_NAME="LSPatchT"
for window_look_back in 48 96 192 336 720; do
    run_experiment $MODEL_NAME $window_look_back 1 1 "weather.csv" "Weather" 64 0.4 12 12
done