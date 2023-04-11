#!/bin/bash


# conda activate sf
#
# python3 train.py --run_name LSTM_Data1_test~28 --dataset ./Data1 --model lstm --lr 0.01 --hidden_dim 16 --num_layers 2 --batch_size 32 # --wandb
# python3 test.py --weights logs/LSTM_Data1_test~28/weights_best.tar --dataset ./Data1 --model lstm --hidden_dim 16 --num_layers 2
#
# python3 train.py --run_name LSTM_Data2_test~28 --dataset ./Data2 --model lstm --lr 0.01 --hidden_dim 16 --num_layers 2 --batch_size 32 # --wandb
# python3 test.py --weights logs/LSTM_Data2_test~28/weights_best.tar --dataset ./Data2 --model lstm --hidden_dim 16 --num_layers 2
#
# python3 train.py --run_name LSTM_Data3_test~8 --dataset ./Data3 --model lstm --lr 0.01 --hidden_dim 16 --num_layers 2 --batch_size 32 # --wandb
# python3 test.py --weights logs/LSTM_Data3_test~8/weights_best.tar --dataset ./Data3 --model lstm --hidden_dim 16 --num_layers 2
#  
# python3 train.py --run_name LSTM_Data4_test~25 --dataset ./Data4 --model lstm --lr 0.01 --hidden_dim 16 --num_layers 2 --batch_size 32 # --wandb
# python3 test.py --weights logs/LSTM_Data4_test~25/weights_best.tar --dataset ./Data4 --model lstm --hidden_dim 16 --num_layers 2
#
#
# b=28
# echo $b
#
# python3 train.py --run_name LSTM_Data5_Batch-$b --dataset ./Data5 --model lstm --lr 0.01 --hidden_dim 16 --num_layers 2 --batch_size 8 # --wandb
# python3 test.py --weights logs/Data5/LSTM_Data5_Batch-$b/weights_best.tar --dataset ./Data5 --model lstm --hidden_dim 16 --num_layers 2
# #
# python3 train.py --run_name RNN_Data5_Batch-$b --dataset ./Data5 --model rnn --lr 0.01 --hidden_dim 16 --num_layers 2 --batch_size 32 # --wandb
# python3 test.py --weights logs/Data5/RNN_Data5_Batch-$b/weights_best.tar --dataset ./Data5 --model rnn --hidden_dim 16 --num_layers 2
#
#
# python data_analysis.py
#
#
for b in 8 11 12 16 22 23 24 25 26 27 28
do
    python plot_results.py --batch $b
done