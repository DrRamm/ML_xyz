# coding: "GBK
from keras.engine.saving import load_model
from numpy import concatenate
from sklearn.metrics import confusion_matrix
import pylab as plt
from common import *

np.random.seed(42)

model = load_model(modelPath)
model.load_weights(modelWeightsPath)


def prepare_all_data():
    print("\n---------------- Prepare ----------------")
    x1, y1 = get_x_y(bad_test, get_ind_by_name("bad"))
    x2, y2 = get_x_y(good_test, get_ind_by_name("good"))
    x3, y3 = get_x_y(avr_folder, get_ind_by_name("avr"))

    full_x = concatenate((x1, x2, x3))
    full_y = concatenate((y1, y2, y3))

    return full_x, full_y


def run_test():
    x_test, y_test = prepare_all_data()
    y_pred = model.predict(x_test)

    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(cm)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


run_test()
print("---------------- Done ----------------")
