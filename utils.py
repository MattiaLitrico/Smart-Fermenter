import numpy as np
import pandas as pd
from os.path import join
from numpy.lib.stride_tricks import sliding_window_view
import math
import pdb
import matplotlib.pyplot as plt
import torch
import os
from sklearn.metrics import r2_score


def load_data(
    work_dir="Data", fermentation_number=8, data_file="data.xlsx", x_cols=[], y_cols=[]
):
    # Load data from .xlsx files

    var_to_keep = [*x_cols, *y_cols]
    data = (
        pd.read_excel(
            io=join(work_dir, str(fermentation_number), data_file), usecols=var_to_keep
        )
        .dropna(how="all")
        .to_dict()
    )

    X = np.zeros((len(data[x_cols[0]].values()), len(x_cols)))
    Y = np.zeros((len(data[x_cols[0]].values()), len(y_cols)))

    if x_cols is not []:
        for idx, x_c in enumerate(x_cols):
            tmp = np.array(list(data[x_c].values()))
            # if x_c in [
            #     # "m_ph",
            #     # "m_stirrer",
            #     "dm_o2",
            #     "dm_air",
            #     "dm_spump1",
            #     "dm_spump2",
            #     "dm_spump3",
            #     "dm_spump4",
            # ]:
            #     # pdb.set_trace()
            #     print(
            #         "################### Normalizing: {} of batch-{} [index-0:{} min:{} and max:{}]".format(
            #             x_c,
            #             str(fermentation_number),
            #             str(tmp[0]),
            #             str(min(tmp)),
            #             str(max(tmp)),
            #         )
            #     )
            #     tmp -= tmp[0]  # min(tmp)
            #     # tmp /= max(tmp)
            #     print("Now: min:{} and max:{}".format(str(min(tmp)), str(max(tmp))))
            X[:, idx] = tmp

    if y_cols is not []:
        for idx, y_c in enumerate(y_cols):
            Y[:, idx] = np.array(list(data[y_c].values()))

    return np.array(X), np.array(Y)


def cumulative2snapshot(data):
    # Trasform cumulative data to snapshot data

    tmp_data = np.insert(data, 0, data[0])[:-1]

    return data - tmp_data


def get_norm_param(X, x_cols=None):
    # Return normalization parameters

    if x_cols is None:
        print("X columns must be defined!")
        return

    return np.mean(X, axis=0), np.std(X, axis=0)


def z_score(x, mean, std, binary_var=None):
    # Compute z-score using mean and standard deviation

    mean = np.repeat(np.expand_dims(mean, axis=0), repeats=x.shape[0], axis=0)
    std = np.repeat(np.expand_dims(std, axis=0), repeats=x.shape[0], axis=0)

    # Remove normalisation for binary var
    mean[:, binary_var] = 0
    std[:, binary_var] = 1

    return (x - mean) / std


def normalise(x, mean=None, std=None, mode="z-score", binary_var=None):
    # Normalise data

    if mode == "z-score":
        if mean is None or std is None:
            print("Mean and std must be defined!")
            return

        return z_score(x, mean, std, binary_var)


def data2sequences(x, ws=20, stride=10):
    # Trasform data into sequences, e.g. [number of sequence, number of features, window size]

    x = np.pad(
        array=x,
        pad_width=((0, compute_padded_length(len(x), ws, stride)), (0, 0)),
        mode="edge",
    )
    sequences = sliding_window_view(x, window_shape=ws, axis=0)[::stride]

    return np.transpose(sequences, (0, 2, 1))


def compute_padded_length(initial_length, ws=20, stride=10):
    # Compute the number of padding elements that needs to be added

    nsw = math.ceil(((initial_length - (ws - 1) - 1) / stride) + 1)
    padded_length = nsw * stride + ws - stride

    return padded_length - initial_length


def polynomial_interpolation(data):
    # Compute polynomial interpolation

    x = np.arange(len(data))
    idx_not_nan = np.argwhere(~np.isnan(data))[:, 0]

    p = np.poly1d(np.polyfit(idx_not_nan, data[idx_not_nan], 6))
    interpolated_y = p(x)
    interpolated_y[interpolated_y < 0] = 0

    data = interpolated_y
    return data


def linear_local_interpolation(data):
    # Compute linear local interpolation

    idx_not_nan = np.argwhere(~np.isnan(data))[:, 0]
    idx_nan = np.argwhere(np.isnan(data))[:, 0]

    interpolated_y = np.interp(idx_nan, idx_not_nan, data[idx_not_nan])

    data[idx_nan] = interpolated_y
    return data


def mix_interpolation(data):
    a, b = (0.5, 0.5)

    return a * linear_local_interpolation(data) + b * polynomial_interpolation(data)


def plot_od600_curve(preds, labels, dir, mae, fpe):
    # Plot predicted values vs real values

    plt.figure(0)
    plt.title("%s_MAE=%.2f_FPE=%.2f%%" % (dir[5:], mae, fpe))
    plt.plot(preds, label="Predicted")
    plt.plot(labels, label="Real interpolated")
    plt.legend()
    plt.xlabel("sample index")
    plt.ylabel("od600")
    plt.savefig(join(dir, "od600pred.png"))

    idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    idx = [int(len(labels) / 20) * x for x in idx]
    idx[-1] = len(labels) - 1
    # print(idx)
    plt.figure(1)
    plt.title("%s_MAE=%.2f_FPE=%.2f%%" % (dir[5:], mae, fpe))
    plt.plot(idx, preds[idx], label="Predicted")
    plt.plot(idx, labels[idx], label="Real interpolated")
    plt.legend()
    plt.xlabel("sample index")
    plt.ylabel("od600")
    plt.savefig(join(dir, "od600pred_10points.png"))


def reject_outliers(data, m=2):
    # Exclude outliers

    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.0
    return data[s < m]


def save_weights(model, e, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    torch.save({"epochs": e, "weights": model.state_dict()}, filename)


def load_checkpoint(weights, cpu=False):
    if not cpu:
        checkpoint = torch.load(weights)
    else:
        checkpoint = torch.load(weights, map_location=torch.device("cpu"))

    return checkpoint


def load_weights(model, weights):
    checkpoint = load_checkpoint(weights=weights)
    model.load_state_dict(checkpoint["weights"])

    return model
