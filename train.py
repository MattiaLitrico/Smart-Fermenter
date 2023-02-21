import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import argparse
import numpy as np
import wandb
from dataset import *
import pdb
import warnings
from model import *
import random

parser = argparse.ArgumentParser(description="PyTorch CIFAR Training")
parser.add_argument("--batch_size", default=256, type=int, help="train batchsize")
parser.add_argument(
    "--lr", "--learning_rate", default=0.02, type=float, help="initial learning rate"
)
parser.add_argument("--hidden_dim", default=32, type=int)
parser.add_argument("--num_layers", default=1, type=int)
parser.add_argument("--num_epochs", default=300, type=int)
parser.add_argument("--seed", default=123)
parser.add_argument("--gpuid", default=0, type=int)
parser.add_argument("--run_name", type=str)
parser.add_argument("--model", default="lstm", type=str)
parser.add_argument("--dataset", type=str, default="")
parser.add_argument("--wandb", action="store_true", help="Use wandb")

args = parser.parse_args()
warnings.filterwarnings("ignore")

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.wandb:
    wandb.init(project="Smart Fermenter", name=args.run_name)


def train(epoch, model, optimiser, trainloader):
    model.train()

    num_iter = (len(trainloader.dataset) // trainloader.batch_size) + 1
    total_loss = 0
    total_err = 0
    iter = 0

    for batch_idx, (input, label) in enumerate(trainloader):
        iter += 1

        batch_size = input.size(0)
        input, label = input.cuda(), label.cuda()

        # Initialise model hidden state
        h = model.init_hidden(batch_size)
        h = tuple([e.data for e in h])

        output, h = model(input.float(), h)

        # Compute loss and error
        loss = compute_loss(output, label)
        # err = mae(output, label.float())
        err = torch.sqrt(mse(output, label.float()))

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        total_loss += loss.item()
        total_err += err.item()

        if iter % 100 == 0:
            sys.stdout.write("\r")
            sys.stdout.write(
                "Epoch [%3d/%3d] Iter[%3d/%3d]\t Loss: %.4f"
                % (
                    epoch,
                    args.num_epochs,
                    batch_idx + 1,
                    num_iter,
                    loss.item(),
                )
            )
            sys.stdout.flush()

    if args.wandb:
        wandb.log(
            {
                "train_loss": total_loss / (iter),
                "train_err": total_err / (iter),
            },
            step=epoch,
        )


def test(epoch, model, testloader):
    model.eval()

    loss = 0
    err = 0

    num_iter = (len(testloader.dataset) // testloader.batch_size) + 1
    iter = 0

    h = model.init_hidden(args.batch_size)

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(testloader):
            iter += 1

            batch_size = input.size(0)
            input, label = input.cuda(), label.cuda()

            # Initialise model hidden state
            h = model.init_hidden(batch_size)
            h = tuple([e.data for e in h])

            output, h = model(input.float(), h)

            # Compute loss and error
            loss += compute_loss(output, label)
            # err += mae(output, label.float()).item()
            err += torch.sqrt(mse(output, label.float())).item()

    loss = loss / iter
    err = err / iter

    sys.stdout.write("\r")
    sys.stdout.write(
        "Epoch [%3d/%3d] Iter[%3d/%3d]\t Loss: %.4f RMSE: %.4f"
        % (epoch, args.num_epochs, batch_idx + 1, num_iter, loss.item(), err)
    )
    sys.stdout.flush()

    if args.wandb:
        wandb.log(
            {
                "val_loss": loss,
                "val_err": err,
            },
            step=epoch,
        )

    return err


def compute_loss(output, label):
    # Compute loss -> MSE
    return mse(output, label.float())


# Setting data
train_dataset = FermentationData(
    work_dir=args.dataset, train_mode=True, y_var=["od_600"]
)
test_dataset = FermentationData(
    work_dir=args.dataset, train_mode=False, y_var=["od_600"]
)

print("Loading training-set!")
trainloader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=args.batch_size, num_workers=2, shuffle=True
)
print("Loading testing-set!")
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=args.batch_size, num_workers=2, shuffle=False
)

# Setting model
if args.model == "lstm":
    model = LSTMPredictor(
        input_dim=train_dataset.get_num_features(),
        hidden_dim=args.hidden_dim,
        output_dim=1,
        n_layers=args.num_layers,
    )
elif args.model == "rnn":
    model = RNNpredictor(
        input_dim=train_dataset.get_num_features(),
        hidden_dim=args.hidden_dim,
        output_dim=1,
        n_layers=args.num_layers,
    )
model.cuda()

mse = nn.MSELoss()
# mae = nn.L1Loss()
optimiser = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0)

# Initialise best rmse as maximum
best = sys.maxsize

# Training
for epoch in range(args.num_epochs + 1):
    print("\nTrain Net")
    train(epoch, model, optimiser, trainloader)

    print("\nTest Net")
    rmse = test(epoch, model, test_loader)

    if rmse < best:
        utils.save_weights(model, epoch, "logs/" + args.run_name + "/weights_best.tar")
        best = rmse
        print("Saving best!")

        if args.wandb:
            wandb.run.summary["best_rmse"] = best
