import numpy as np
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer

rows = 3
chunk_size = 700
model = Sequential()

GOOD_NUMBER = int(1)
BAD_NUMBER = int(2)
AVR_NUMBER = int(3)

labels = ["good", "average", "bad"]
encoder = LabelBinarizer()
encoder.fit_transform(labels)
y_labels = encoder.classes_


modelPath = "tr_model.h5"
modelWeightsPath = 'tr_weights.h5'
folder = "./txt_data/"

file_good = folder + "good_full.txt"
file_good_cut = folder + "good_cut.txt"

file_bad = folder + "bad_full.txt"
file_bad_cut = folder + "bad_cut.txt"

file_avr = folder + "average.txt"
file_avr_cut = folder + "average_cut.txt"


def score(X_test):
    yhat = model.predict(X_test, verbose=1)
    return yhat


def get_data_from_file(file):
    print("Open file %s" % file)
    text_file = open(file, "r")
    data_list = text_file.read().split("\n")

    lines = int(len(data_list)/rows)
    print("Got %d lines" % lines)
    if lines % chunk_size:
        print("trimming data")
        trim_size = (lines % chunk_size) * rows
        data_list = data_list[trim_size:]
    data_healthy = np.asarray(data_list, dtype=np.float64).reshape((-1, rows))

    return data_healthy


def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data)


def reshape_data(data):
    print("Input data shape is ", data.shape)
    chunks = int(len(data) / chunk_size)
    data.shape = (chunks, chunk_size, rows)
    print("X shaped to ", data.shape)
    return data, chunks
