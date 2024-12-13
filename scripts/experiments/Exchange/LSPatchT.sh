#export CUDA_VISIBLE_DEVICES=1
INPUT_LENGTH=96

# building the pretrained model fpr LSPatchT
echo "Building the pretrained model for LSPatchT for Weather dataset with input length $INPUT_LENGTH"
python -u experiment.py \
    --is_pretrain 1 \
    --is_training 1 \
    --batch_size 64 \
    --root_path dataset/ \
    --data_path exchange_rate.csv \
    --checkpoints ./experiments/model_saved/checkpoints/ \
    --model LSPatchT \
    --model_id Pretrain \
    --data Exchange \
    --freq h \
    --features M \
    --mask_ratio 0.4 \
    --patch_len 12 \
    --stride 12 \
    --seq_len $INPUT_LENGTH \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des Exp \
    --itr 1

for i in 24 48 96 168 192 336 720; do
  echo "Starting to train model with output length = $i"
  python -u experiment.py \
    --is_training 1 \
    --is_finetune 1 \
    --batch_size 32 \
    --root_path dataset/ \
    --data_path exchange_rate.csv \
    --model LSPatchT \
    --model_id FineTune \
    --data Exchange \
    --checkpoints ./experiments/model_saved/checkpoints/ \
    --pretrained_model ./experiments/model_saved/checkpoints/Pretrain_LSPatchT_Exchange_sl96_0/checkpoint.pth \
    --freq h \
    --features M \
    --seq_len $INPUT_LENGTH \
    --pred_len $i \
    --patch_len $INPUT_LENGTH \
    --stride $INPUT_LENGTH \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des Exp \
    --itr 2
  echo "Finished training model with output length = $i"
  echo "=================================================="
done
