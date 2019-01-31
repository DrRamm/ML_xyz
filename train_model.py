# coding: "GBK
from common import *
from keras.callbacks import EarlyStopping

from keras.layers import LSTM, Dense, GRU, Dropout, Flatten, Conv2D, MaxPooling2D, TimeDistributed, Embedding, Conv1D, \
    MaxPooling1D
from keras.callbacks import TensorBoard

np.random.seed(42)


def create_model():
    input_shape = (chunk_size, rows)

    # model.add(Conv1D(32, 3, activation='relu'))
    # model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(units=30, input_shape=input_shape, return_sequences=True))
    # model.add(LSTM(units=50, input_shape=input_shape))
    model.add(LSTM(units=30))
    model.add(Dropout(0.15))
    model.add(Dense(rows,  activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',  metrics=['accuracy'])


early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
tensor_board = TensorBoard(log_dir='./Graph_training', histogram_freq=2000,
                           write_graph=True, write_images=True, write_grads=True)

callback = [tensor_board, early_stop]


def fit_model(x, y):
    model.fit(x, y, epochs=5, validation_split=0.3, verbose=0, shuffle=True, callbacks=callback)


#######################


def train_model(file, number):
    print("---------------- Train %d ----------------" % number)
    data_scaled = scale_data(get_data_from_file(file))
    data_reshaped, chunks = reshape_data(data_scaled)
    y = np.full((chunks, 3), number, "int32")
    print("Y shaped to ", y.shape)

    for i in range(2):
        print("---------------- %d ----------------" % i)
        fit_model(data_reshaped, y)


def check_acc(x_test, y_test):
    loss, acc = model.evaluate(x_test, y_test)
    print("Model loss ", loss)
    print("Model acc %f %%" % (acc*100))


def check_model(file, number):
    print("---------------- Check %d for file %s ----------------" % (number, file))
    data_scaled = scale_data(get_data_from_file(file))
    data_reshaped, chunks = reshape_data(data_scaled)
    y = np.full((chunks, 3), number, "int32")
    print("Y shaped to ", y.shape)

    check_acc(data_reshaped, y)


create_model()

train_model(file_good, GOOD_NUMBER)
train_model(file_bad, BAD_NUMBER)
train_model(file_avr, AVR_NUMBER)

check_model(file_good_cut, GOOD_NUMBER)
check_model(file_bad_cut,  BAD_NUMBER)
check_model(file_avr_cut,  AVR_NUMBER)

print("Saving model")
model.save(modelPath)
model.save_weights(modelWeightsPath)

print("---------------- Done ----------------")
print("tensorboard --logdir ./Graph_training --host=127.0.0.1")
