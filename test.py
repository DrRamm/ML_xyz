# coding: "GBK
from keras.engine.saving import load_model
from common import *

np.random.seed(42)

model = load_model(modelPath)
model.load_weights(modelWeightsPath)


def run_test(file_name):
    print("---------------- Test %s ----------------" % file_name)
    data_scaled = scale_data(get_data_from_file(file_name))
    data_reshaped, chunks = reshape_data(data_scaled)
    predictions = model.predict(data_reshaped)
    return predictions.argmax()


good_pred = run_test(file_good_cut)
bad_pred = run_test(file_bad_cut)
avr_pred = run_test(file_avr_cut)

print("good - %d" % good_pred)
print("bad  - %d" % bad_pred)
print("avr  - %d" % avr_pred)
print("---------------- Done ----------------")
