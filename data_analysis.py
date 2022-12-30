import utils
import numpy as np
import pdb
import matplotlib.pyplot as plt

work_dir = "Data1/"
# work_dir = "Data/"
# work_dir = "Data3/"
# work_dir = "Data4/"

# input variables
x_var = [
    "m_ph",
    # "m_ls_opt_do",
    # "m_temp",
    # "m_stirrer",
    # "dm_o2",
    # "dm_air",
    # "dm_spump1",
    # "dm_spump2",
    # "dm_spump3",
    # "dm_spump4",
    # "induction",
]
y_var = ["od_600"]


def mix_interpolation(data):
    # Compute linear local interpolation
    data = data[:, 0]

    a, b = (0.5, 0.5)

    if np.isnan(data[0]):
        data[0] = 0

    return utils.mix_interpolation(data).reshape(-1, 1)


def data2sequences(self, X, ws=20, stride=10):
    # Transform data to sequences with default sliding window 20 and stride 10
    return utils.data2sequences(X, ws, stride)


def preprocess_labels(Y, norm_mode="z-score", ws=20, stride=10):
    # Preprocess labels
    processed_Y = []
    for y in Y:
        y = mix_interpolation(y)
    #     pdb.set_trace()
    #     y = data2sequences(y, ws, stride)
    #     processed_Y.append(y)

    # processed_Y = np.concatenate(processed_Y, axis=0)

    return y  # processed_Y


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


plt.figure(0)
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
    data = utils.load_data(
        work_dir=work_dir,
        fermentation_number=n,
        data_file="data.xlsx",
        x_cols=x_var,
        y_cols=y_var,
    )
    #
    Y = data[1][data[1] > 0]
    # pdb.set_trace()
    plt.plot(unique(np.where(data[1] == Y)[0]), Y, label="batch-%d" % (n))
    #
    # Y_I = preprocess_labels([data[1]], norm_mode="z-score", ws=20, stride=1)
    # plt.plot(Y_I, label="I_batch-%d" % (n))
    #
    print("batch-%d, first value=%f, last value=%f" % (n, Y[0], Y[1]))

plt.legend()
plt.xlabel("sample index")
plt.ylabel("od600")
plt.title(work_dir[:-1])
plt.savefig("data_od600.jpg")
