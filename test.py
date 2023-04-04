import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import argparse
import numpy as np
from dataset import *
import pdb
import warnings
from model import *
import random
import utils
import math

parser = argparse.ArgumentParser(description="PyTorch CIFAR Training")
parser.add_argument("--batch_size", default=256, type=int, help="test batchsize")
parser.add_argument("--hidden_dim", default=16, type=int)
parser.add_argument("--num_layers", default=2, type=int)
parser.add_argument("--seed", default=123)
parser.add_argument("--gpuid", default=0, type=int)
parser.add_argument("--weights", type=str, default="")
parser.add_argument("--dataset", type=str, default="")
parser.add_argument("--model", default="lstm", type=str)

args = parser.parse_args()
warnings.filterwarnings("ignore")

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Predict with overlapped sequences
def test(epoch, model, testloader):
    model.eval()

    loss = 0
    err = 0

    iter = 0

    # Initialise model hidden state
    h = model.init_hidden(args.batch_size)

    # Initialise vectors to store predictions, labels
    preds = np.zeros(len(test_dataset) + test_dataset.ws - 1)
    labels = np.zeros(len(test_dataset) + test_dataset.ws - 1)
    n_overlap = np.zeros(len(test_dataset) + test_dataset.ws - 1)

    N = 10
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(testloader):
            iter += 1

            batch_size = input.size(0)

            h = model.init_hidden(batch_size)
            h = tuple([e.data for e in h])

            input, label = input.cuda(), label.cuda()

            output, h = model(input.float(), h)
            # pdb.set_trace()

            y = output.view(-1).cpu().numpy()
            y_padded = np.pad(y, (N // 2, N - 1 - N // 2), mode="edge")
            y_smooth = np.convolve(y_padded, np.ones((N,)) / N, mode="valid")

            # Store predictions and labels summing over the overlapping
            preds[
                batch_idx : (batch_idx + test_dataset.ws)
            ] += y_smooth  # (output.view(-1).cpu().numpy())
            labels[batch_idx : (batch_idx + test_dataset.ws)] += (
                label.view(-1).cpu().numpy()
            )
            n_overlap[batch_idx : (batch_idx + test_dataset.ws)] += 1.0

            loss += mse(output, label.float())
            # err += mae(output, label.float()).item()
            err += torch.sqrt(mse(output, label.float())).item()

    loss = loss / len(test_dataset)
    err = err / len(test_dataset)

    # Compute the average dividing for the number of overlaps
    preds /= n_overlap
    labels /= n_overlap

    return err, preds, labels


# Setting data
test_dataset = FermentationData(
    work_dir=args.dataset, train_mode=False, y_var=["od_600"]
)

print("Loading testing-set!")
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=1, num_workers=2, shuffle=False
)

# Setting model
if args.model == "lstm":
    model = LSTMPredictor(
        input_dim=test_dataset.get_num_features(),
        hidden_dim=args.hidden_dim,
        output_dim=1,
        n_layers=args.num_layers,
    )
elif args.model == "rnn":
    model = RNNpredictor(
        input_dim=test_dataset.get_num_features(),
        hidden_dim=args.hidden_dim,
        output_dim=1,
        n_layers=args.num_layers,
    )

model.cuda()

weights = (
    os.path.join("model_weights", "od600_prediction", args.model, "weights_best.tar")
    if args.weights == ""
    else args.weights
)
model = utils.load_weights(model, weights)

mse = nn.MSELoss()
# mae = nn.L1Loss()

# Testing
print("\nTesting")
err, preds, labels = test(0, model, test_loader)

preds = preds.reshape(-1, 1)
labels = labels.reshape(-1, 1)

preds = preds[50:]
labels = labels[50:]


# mae = (abs(preds - labels)).mean()
# fpe = abs(preds[-1] - labels[-1]) / labels[-1] * 100

mse = np.square(np.subtract(preds, labels)).mean()
rmse = math.sqrt(mse)

# Relative Error on Final Yield
refy = abs(preds[-1] - labels[-1]) / labels[-1] * 100

# pdb.set_trace()
np.savez(
    weights.split("/weights")[0] + "/results.npz",
    preds=preds,
    labels=labels,
    rmse=rmse,
    refy=refy,
)
print("Saved: ", weights.split("/weights")[0] + "/results.npz")
#
utils.plot_od600_curve(
    preds, labels, weights[:-17], rmse, refy
)  # remove weights_best.tar from weights path

# print("\nMAE Error OD600: ", mae)
# print("\nFPE: ", fpe)  # , "%")
print("\nRMSE Error OD600: ", rmse)
print(
    "\nREFY: %.2f%%" % (refy), "[absolute error: %.2f]" % (abs(preds[-1] - labels[-1]))
)
