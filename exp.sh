# python3 train.py --run_name LSTM_Data1_test~28 --dataset ./Data1 --model lstm --lr 0.01 --hidden_dim 16 --num_layers 2 --batch_size 32 # --wandb
# python3 test.py --weights logs/LSTM_Data1_test~28/weights_best.tar --dataset ./Data1 --model lstm --hidden_dim 16 --num_layers 2
#
# python3 train.py --run_name LSTM_Data2_test~28 --dataset ./Data2 --model lstm --lr 0.01 --hidden_dim 16 --num_layers 2 --batch_size 32 # --wandb
# python3 test.py --weights logs/LSTM_Data2_test~28/weights_best.tar --dataset ./Data2 --model lstm --hidden_dim 16 --num_layers 2
#
# python3 train.py --run_name LSTM_Data3_test~28 --dataset ./Data3 --model lstm --lr 0.01 --hidden_dim 16 --num_layers 2 --batch_size 32 # --wandb
# python3 test.py --weights logs/LSTM_Data3_test~28/weights_best.tar --dataset ./Data3 --model lstm --hidden_dim 16 --num_layers 2
#
python3 train.py --run_name LSTM_Data4_test~28 --dataset ./Data4 --model lstm --lr 0.01 --hidden_dim 16 --num_layers 2 --batch_size 32 # --wandb
python3 test.py --weights logs/LSTM_Data4_test~28/weights_best.tar --dataset ./Data4 --model lstm --hidden_dim 16 --num_layers 2
#
#
#
# python data_analysis.py