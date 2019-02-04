# coding: "GBK

from keras.optimizers import Adam
from numpy import concatenate

from common import *
from keras.callbacks import EarlyStopping

from keras.layers import LSTM, Dense, GRU, Dropout, Flatten, Conv2D, MaxPooling2D, TimeDistributed, Embedding, Conv1D, \
    MaxPooling1D
from keras.callbacks import TensorBoard

np.random.seed(42)

UNIT = 20
EPOCHS = 8
BATCH = 2
PAT = 10
VAL_SPLIT = 0.2
FILTERS = 32
KERN_SIZE = 3


def create_model():
    input_shape = (chunk_size, rows)

    model.add(Conv1D(FILTERS, KERN_SIZE, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    # model.add(LSTM(units=30, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(units=UNIT, input_shape=input_shape))
    # model.add(LSTM(units=30))
    model.add(Dropout(0.15))
    model.add(Dense(rows,  activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(clipnorm=1.),  metrics=['accuracy'])


graph_name = ("epo=%d batch=%d pat=%d unit=%d vsplit=%.2f filt=%d ksize=%d" % (EPOCHS, BATCH,PAT, UNIT, VAL_SPLIT, FILTERS, KERN_SIZE))

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=PAT, verbose=0, mode='auto')
tensor_board = TensorBoard(log_dir='./Graph_training/' + graph_name, histogram_freq=2000,
                           write_graph=True, write_images=True, write_grads=True)

callback = [tensor_board, early_stop]


def fit_model(x, y):
    for i in range(1):
        model.fit(x, y, epochs=EPOCHS, batch_size=BATCH, validation_split=VAL_SPLIT, verbose=1, shuffle=True, callbacks=callback)


#######################


def prepare_all_data():
    print("\n---------------- Prepare ----------------")
    x1, y1 = get_x_y(bad_folder, get_ind_by_name("bad"))
    x2, y2 = get_x_y(good_folder, get_ind_by_name("good"))
    x3, y3 = get_x_y(avr_folder, get_ind_by_name("avr"))

    full_x = concatenate((x1, x2, x3))
    full_y = concatenate((y1, y2, y3))

    final_x, final_y = shuffle_x_y(full_x, full_y)

    print(final_x)
    print(final_y)

    print("\nPrepared shapes:")
    print("X is ", final_x.shape)
    print("Y is ", final_y.shape)
    return final_x, final_y


def train_model():
    x, y = prepare_all_data()
    fit_model(x, y)
    check_acc(x, y)


def check_acc(x_test, y_test):
    print("\n---------------- Checking evaluate ----------------")
    loss, acc = model.evaluate(x_test, y_test)
    print("Model loss ", loss)
    print("Model acc %f %%" % (acc*100))


create_model()
train_model()

print("\nSaving model")
model.save(modelPath)
model.save_weights(modelWeightsPath)

print("\n---------------- Done ----------------")
print("Look at %s graph" % graph_name)
print("tensorboard --logdir ./Graph_training --host=127.0.0.1")
