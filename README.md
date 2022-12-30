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
python3 train.py --run_name lstm_tr-11-24_ts-8 --wandb --model lstm --lr 0.01 --hidden_dim 16 --num_layers 2 --batch_size 32
or
python3 train.py --run_name rnn_tr-b11-b24_ts-b8 --wandb --model rnn --lr 0.01 --hidden_dim 16 --num_layers 2 --batch_size 256
```

## Testing
```
python3 test.py --weights logs/lstm_tr-b22-b28_ts-b8/weights_best.tar --model lstm --hidden_dim 16 --num_layers 2
or
python3 test.py --weights logs/rnn_tr-b11-b24_ts-b8/weights_best.tar --model rnn --hidden_dim 16 --num_layers 2 
```

