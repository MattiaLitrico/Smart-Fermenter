# Smart-Fermenter

## Create Environment 
```
conda create -y -n sf
```

# Install Dependencies
```
conda activate sf
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install wandb --channel conda-forge
conda install -c anaconda pandas openpyxl
conda install -c conda-forge matplotlib scikit-learn
```

## Training
```
python3 train.py --run_name LSTM_Data5_Batch-28 --dataset ./Data5 --model lstm --lr 0.01 --hidden_dim 16 --num_layers 2 --batch_size 8 # --wandb
or
python3 train.py --run_name RNN_Data5_Batch-28 --dataset ./Data5 --model rnn --lr 0.01 --hidden_dim 16 --num_layers 2 --batch_size 32 # --wandb
```

## Testing
```
python3 test.py --weights logs/Data5/LSTM_Data5_Batch-28/weights_best.tar --dataset ./Data5 --model lstm --hidden_dim 16 --num_layers 2
or
python3 test.py --weights logs/Data5/RNN_Data5_Batch-28/weights_best.tar --dataset ./Data5 --model rnn --hidden_dim 16 --num_layers 2 
```

