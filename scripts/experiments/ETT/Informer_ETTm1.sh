#export CUDA_VISIBLE_DEVICES=1

# ETTm1 dataset
for i in 24 48 96 168 192 336 720; do
  echo "Starting to train model with output length = $i"
  python -u experiment.py \
    --is_training 1 \
    --batch_size 32 \
    --root_path dataset/ETT/ \
    --data_path ETTm1.csv \
    --model Informer \
    --data ETTm1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $i \
    --e_layers 2 \
    --d_layers 1 \
    --factor 5 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des Exp \
    --freq t \
    --itr 2

  echo "Finished training model with output length = $i"
  echo "=================================================="
done
