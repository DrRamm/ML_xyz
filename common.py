import os

import numpy as np
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer

rows = 3
chunk_size = 2
model = Sequential()

labels = ["good", "average", "bad"]
encoder = LabelBinarizer()
encoder.fit_transform(labels)


def get_ind_by_name(name):
    return encoder.transform([name]).argmax()


modelPath = "tr_model.h5"
modelWeightsPath = 'tr_weights.h5'

train_folder = "./train_data_2/"
test_folder = "./test_data/"

good_folder = train_folder + "good" + "/"
bad_folder = train_folder + "bad" + "/"
avr_folder = train_folder + "average" + "/"

good_test = test_folder + "good" + "/"
bad_test = test_folder + "bad" + "/"
avr_test = test_folder + "average" + "/"


def file_to_list(path):
    print("Open file %s" % path)
    text_file = open(path, "r")
    return text_file.read().split("\n")


def trim_list(list_name):
    lines = int(len(list_name) / rows)
    print("Got %d lines" % lines)
    if lines % chunk_size:
        print("trimming data")
        trim_size = (lines % chunk_size) * rows
        list_name = list_name[trim_size:]
    return list_name


def get_data_from_folder(folder):
    data_list = []
    for file in os.listdir(folder):
        data_list.extend(file_to_list(folder + file))
        break

    data_list = trim_list(data_list)
    return np.asarray(data_list, dtype=np.float64).reshape((-1, rows))


def get_data_from_file(path):
    data_list = file_to_list(path)
    data_list = trim_list(data_list)
    return np.asarray(data_list, dtype=np.float64).reshape((-1, rows))


def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data)


def reshape_data(data):
    print("Input data shape is ", data.shape)
    chunks = int(len(data) / chunk_size)
    data.shape = (chunks, chunk_size, rows)
    print("X shaped to ", data.shape)
    return data, chunks


def get_x_y(folder, number):
    print("\n---------------- get XY for %d ----------------" % number)
    data_scaled = scale_data(get_data_from_folder(folder))
    x, chunks = reshape_data(data_scaled)
    y = np.eye(rows)[[number] * chunks]
    print(x)
    print(y)
    print("Y shaped to ", y.shape)
    return x, y


def split_array_to_list(input_array):
    output_list = []
    for ind in range(len(input_array)):
        temp = input_array[ind, :].tolist()
        temp2 = temp[0].tolist()
        for indR in range(len(temp2)):
            output_list.append(temp2[indR])
    return output_list


def shuffle_x_y(x, y):
    temp_array = []
    for ind in range(len(y)):
        temp_array.append(x[ind])
        temp_array.append(y[ind])

    full_x_y = np.asarray(temp_array, dtype=object).reshape((-1, 2))
    np.random.shuffle(full_x_y)
    temp_x, temp_y = np.hsplit(full_x_y, 2)

    x_list = split_array_to_list(temp_x)
    y_list = split_array_to_list(temp_y)

    final_y = np.asarray(y_list, dtype=np.float64).reshape((-1, rows))
    final_x = np.asarray(x_list, dtype=np.float64).reshape((len(final_y), -1, rows))

    return final_x, final_y
