model_name=Transformer
data_name=ETTh2.csv
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path $data_name \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --itr 1 \

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path $data_name \
  --model_id ETTh2_96_192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --itr 1 \

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path $data_name \
  --model_id ETTh2_96_384 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 384 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path $data_name \
  --model_id ETTh2_96_768 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 768 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \