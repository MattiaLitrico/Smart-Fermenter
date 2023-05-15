import utils
import numpy as np
import pdb
import matplotlib.pyplot as plt
import os
from scipy import signal

# input variables
x_var = [
    "m_ph",
    "m_ls_opt_do",
    "m_temp",
    "m_stirrer",
    "dm_o2",
    "dm_air",
    "dm_spump1",
    "dm_spump2",
    "dm_spump3",
    "dm_spump4",
    "induction",
]
y_var = ["od_600"]


def mix_interpolation(data):
    # Compute linear local interpolation
    data = data[:, 0]

    a, b = (0.5, 0.5)

    if np.isnan(data[0]):
        data[0] = 0

    return utils.mix_interpolation(data).reshape(-1, 1)


def data2sequences(X, ws=20, stride=10):
    # Transform data to sequences with default sliding window 20 and stride 10
    return utils.data2sequences(X, ws, stride)


def preprocess_labels(Y, norm_mode="z-score", ws=20, stride=10):
    # Preprocess labels
    processed_Y = []
    for y in Y:
        y = mix_interpolation(y)
        y = data2sequences(y, ws, stride)
        processed_Y.append(y)

    processed_Y = np.concatenate(processed_Y, axis=0)

    return processed_Y


def get_interpolation(Y, norm_mode="z-score", ws=20, stride=10):
    # Preprocess labels
    y_int = mix_interpolation(Y)
    processed_Y = data2sequences(y_int, ws, stride)

    return y_int, processed_Y


def unique(list1):

    # initialize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    #
    return unique_list


### plot x_var
work_dir = "Data5/"

results_dir = "data_analysis/" + work_dir[:-1]

if not os.path.exists(results_dir):
    os.makedirs(results_dir)


for var in range(len(x_var)):
    plt.figure(var) #,figsize=(10, 5))
    for n in [
        8,
        11,
        12,
        # 14,
        16,
        # 17,
        # 19,
        # 20,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
    ]:
        # pdb.set_trace()
        data = utils.load_data(
            work_dir=work_dir,
            fermentation_number=n,
            data_file="data.xlsx",
            x_cols=[x_var[var]],
            y_cols=y_var,
        )
        #
        X = data[0]  # [data[1] > 0]
        if x_var[var]=="m_stirrer" or x_var[var]=="m_ls_opt_do": 
            scale_factor = 10
            # X =  signal.resample(X, int(len(X)/scale_factor)) # downsampling
            # x1=np.arange(len(X))*scale_factor
            x1=list(range(0, len(X), scale_factor))
            plt.plot(x1,X[x1], label="batch-%d" % (n), linewidth=2)
        else:
            plt.plot(X, label="batch-%d" % (n), linewidth=2)
        #
        # Y_I = preprocess_labels([data[1]], norm_mode="z-score", ws=20, stride=1)
        # plt.plot(Y_I, label="I_batch-%d" % (n))
        #
        print(
            "Var: %s, batch-%d, first value=%f, last value=%f"
            % (x_var[var], n, X[0], X[-1])
        )

    plt.legend(ncol=3)
    plt.xlabel("timestamp", fontsize=14)
    plt.ylabel(x_var[var], fontsize=14)
    plt.tight_layout()
    # plt.autoscale(enable=True, axis='x', tight=True)
    # plt.title(work_dir[:-1])
    plt.savefig("data_analysis/" + work_dir[:-1] + "/" + x_var[var] + ".png", dpi=600)
    plt.close()


##################################################################################
# pdb.set_trace()
work_dir = "Data1/"
results_dir = "data_analysis/"
### plot all od600
plt.figure() #figsize=(10, 5))
for n in [
    8,
    11,
    12,
    14,
    16,
    17,
    19,
    20,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
]:
    Y = []

    data = utils.load_data(
        work_dir=work_dir,
        fermentation_number=n,
        data_file="data.xlsx",
        x_cols=x_var,
        y_cols=y_var,
    )
    Y.append(data[1])

    Y2 = np.array(Y)

    # #
    # Y = data[1][data[1] > 0]
    # plt.loglog(unique(np.where(data[1] == Y)[0]), Y, label="batch-%d" % (n))
    #
    # Y_I = preprocess_labels([data[1]], norm_mode="z-score", ws=20, stride=1)
    # plt.plot(Y_I, label="I_batch-%d" % (n))
    #
    # pdb.set_trace()
    Y_I = preprocess_labels(Y2, norm_mode="z-score", ws=20, stride=1)
    Y3 = Y_I[:, -1, -1]
    plt.semilogy(Y3, label="batch-%d" % (n), linewidth=2)
    #
    print("Var: od600, batch-%d, first value=%f, last value=%f" % (n, Y3[0], Y3[-1]))

plt.legend(ncol=3)
plt.xlabel("timestamp", fontsize=14)
plt.ylabel("OD$_{600nm}$", fontsize=14)
plt.tight_layout()
# plt.title(work_dir[:-1])

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

plt.savefig(results_dir + "/" + "all-od600.png", dpi=600)


##################################################################################
# pdb.set_trace()
work_dir = "Data5/"
results_dir = "data_analysis/"
### plot all od600
plt.figure() #figsize=(10, 5))
for n in [
    8,
    11,
    12,
    16,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
]:
    Y = []

    data = utils.load_data(
        work_dir=work_dir,
        fermentation_number=n,
        data_file="data.xlsx",
        x_cols=x_var,
        y_cols=y_var,
    )
    Y.append(data[1])

    Y2 = np.array(Y)

    # #
    # Y = data[1][data[1] > 0]
    # plt.loglog(unique(np.where(data[1] == Y)[0]), Y, label="batch-%d" % (n))
    #
    # Y_I = preprocess_labels([data[1]], norm_mode="z-score", ws=20, stride=1)
    # plt.plot(Y_I, label="I_batch-%d" % (n))
    #
    # pdb.set_trace()
    Y_I = preprocess_labels(Y2, norm_mode="z-score", ws=20, stride=1)
    Y3 = Y_I[:, -1, -1]
    plt.semilogy(Y3, label="batch-%d" % (n), linewidth=2)
    #
    print("Var: od600, batch-%d, first value=%f, last value=%f" % (n, Y3[0], Y3[-1]))

plt.legend(ncol=3)
plt.xlabel("timestamp", fontsize=14)
plt.ylabel("OD$_{600nm}$", fontsize=14)
plt.tight_layout()
# plt.title(work_dir[:-1])

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

plt.savefig(results_dir + "/" + "selected-od600.png", dpi=600)


##################################################################################
# pdb.set_trace()
work_dir = "Data5/"
results_dir = "data_analysis/"
### plot
plt.figure() #figsize=(10, 5))
Y = []
n = 8

data = utils.load_data(
    work_dir=work_dir,
    fermentation_number=n,
    data_file="data.xlsx",
    x_cols=x_var,
    y_cols=y_var,
)
Y = data[1]

newList = []
newIndex = []
idx = 0
for element in Y:
    if str(element[0]) != "nan":
        newList.append(element[0])
        newIndex.append(idx)
    idx += 1

Y = np.array(Y)


y_int, y = get_interpolation(Y, norm_mode="z-score", ws=20, stride=10)
plt.plot(
    newIndex,
    newList,
    "r",
    label="actual values",
    marker="x",
    markersize=10,
    linewidth=0,
)
plt.plot(y_int, label="interpolated", linewidth=2)

plt.legend(ncol=3)
plt.xlabel("timestamp", fontsize=14)
plt.ylabel("OD$_{600nm}$", fontsize=14)
plt.tight_layout()
# plt.title(work_dir[:-1])

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

plt.savefig(results_dir + "/" + "interpolation.png", dpi=600)
