#export CUDA_VISIBLE_DEVICES=1

for i in 24 48 96 168 192 336 720; do
  echo "Starting to train model with output length = $i"
  python -u experiment.py \
    --is_training 1 \
    --batch_size 32 \
    --root_path dataset/ \
    --data_path exchange_rate.csv \
    --model Crossformer \
    --data Exchange \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $i \
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
