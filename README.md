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
python3 train.py --model lstm --hidden_dim 16 --num_layers 2 --batch_size 64 --run_name od600_lstm_test-b8 --wandb
```

## Testing
```
python3 test.py --model lstm --hidden_dim 16 --num_layers 2 --weights logs/od600_lstm_test-b8/weights_best.tar
```