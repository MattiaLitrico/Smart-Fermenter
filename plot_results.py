import numpy as np
import matplotlib.pyplot as plt
import pdb

import argparse


# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-b", "--batch", help="Batch number")

# Read arguments from command line
args = parser.parse_args()


def save_plot(file_path, lstm, rnn):
    #
    plt.figure()
    # pdb.set_trace()
    plt.semilogy(
        lstm["preds"],
        label="LSTM",  # (RMSE:%.2f, REFY:%.2f)" % (lstm["rmse"], lstm["refy"]),
        linewidth=2,
    )
    plt.semilogy(
        rnn["preds"],
        label="RNN",  #  (RMSE:%.2f, REFY:%.2f)" % (rnn["rmse"], rnn["refy"]),
        linewidth=2,
    )
    plt.semilogy(rnn["labels"], label="Ground truth", linestyle="--", linewidth=2)
    plt.legend(loc="lower right")
    # plt.ylim(bottom=0.5)
    # plt.grid(which="minor")
    plt.xlabel("samples", fontsize=14)
    plt.ylabel("OD$_{600nm}$", fontsize=14)
    plt.tight_layout()
    # plt.title("Batch-" + file_path.split("/")[-1])
    # text on plot
    # plt.text(500, 2, "This text starts at point (2,4)", horizontalalignment="right")
    plt.savefig(file_path + ".png", dpi=600)
    print(
        "Saved: ",
        file_path + ".png",
        "[REFY: %.2f%%,  %.2f%%"
        % (
            abs(lstm["preds"][-1] - lstm["labels"][-1]) / lstm["labels"][-1] * 100,
            abs(rnn["preds"][-1] - rnn["labels"][-1]) / rnn["labels"][-1] * 100,
        ),
        ",",
        rnn["labels"][-1],
        "]",
    )


# args.batch = 8
dir_path = "logs/Data5/"
LSTM_path = dir_path + "LSTM_Data5_Batch-" + str(args.batch)
RNN_path = dir_path + "RNN_Data5_Batch-" + str(args.batch)
LSTM_data = np.load(LSTM_path + "/results.npz")
RNN_data = np.load(RNN_path + "/results.npz")
save_plot(dir_path + str(args.batch), LSTM_data, RNN_data)
