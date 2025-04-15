export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer

root_path='C:/my/硕士/科研数据/all_datasets/ETT-small/'seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=10
patience=10
batch_size=16

python -u run.py --task_name long_term_forecast --is_training 1 --root_path C:/my/硕士/科研数据/all_datasets/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_96_96 --model TimeMixer --data ETTh1 --features MS --seq_len 96 --label_len 0 --pred_len 96 --e_layers 2 --enc_in 7 --c_out 7 --des 'Exp' --itr 1 --d_model 16 --d_ff 32 --learning_rate 0.01 --train_epochs 10 --patience 10 --batch_size 128 --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2
python -u run.py --task_name noise_forecast --is_training 1 --root_path C:/my/硕士/科研数据/all_datasets/Noise/ --data_path Noise_1000Hz_4896.csv --model_id Noise_96_96 --model TimeMixer --data NOISE4896 --features MS --seq_len 96 --label_len 0 --pred_len 96 --e_layers 2 --enc_in 6 --c_out 6 --des 'Exp' --itr 1 --d_model 16 --d_ff 32 --learning_rate 0.01 --train_epochs 10 --patience 10 --batch_size 128 --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  $root_path\
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --e_layers $e_layers \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 128 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 192 \
  --e_layers $e_layers \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 128 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 336 \
  --e_layers $e_layers \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 128 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 720 \
  --e_layers $e_layers \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 128 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window
