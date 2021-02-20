from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
tf.enable_eager_execution()
import numpy as np

import pandas as pd
import yaml
import autoencoder_model
from keras import optimizers, Sequential
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint, TensorBoard
import neural_network_evaluator
import visualiser
import keras
import seaborn as sns
import pickle
sns.set()
# import neural_network_evaluator
# import visualiser

file_to_read = open('..\\pickle\\preprocessed_dataset_ann.pickle', "rb")
# file_to_read = open('/home/mohammed/pickle/preprocessed_dataset_ann_n500.pickle', "rb")
loaded_object = pickle.load(file_to_read)
file_to_read.close()
# print(loaded_object)
dataset = loaded_object

with open("initializer.yaml") as stream:
    param = yaml.safe_load(stream)

threshold_fixed = param["threshold_fixed"]
LABELS = param["LABEL"]
datatset = dataset["n_result"]

df_train, df_test = train_test_split(datatset,
                                     test_size=param["data_split"]["test_size"],
                                     shuffle=param["data_split"]["shuffle"],
                                     random_state=param["data_split"]["random_state"])

df_train, df_valid = train_test_split(df_train,
                                      test_size=param["data_split"]["test_size"],
                                      random_state=param["data_split"]["random_state"])
x_test_normal = dataset["normal_data"].iloc[0:dataset["abnormal_data"].shape[0]]
train = dataset["normal_data"].iloc[dataset["abnormal_data"].shape[0]:].reset_index(drop=True)
print("x_test_normal : ", x_test_normal)
test = pd.concat([x_test_normal, dataset["abnormal_data"]], axis=0, ignore_index=True)
print("x_train : ", train)
print("x_test : ", test)
print(train.shape, test.shape)

TIME_STEPS = 5


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


print(test)
# reshape to [samples, time_steps, n_features]
x_train, y_train = create_dataset(
  train[['label']],
  train.label,
  TIME_STEPS
)
x_test, y_test = create_dataset(
  test[['label']],
  test.label,
  TIME_STEPS
)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# print(y_test)
input_dim1 = x_train.shape[1]
input_dim2 = x_train.shape[2]
print(input_dim1)
print(input_dim2)
# ######################################

timesteps = x_train.shape[1] # equal to the lookback
n_features = x_train.shape[2] # 59
lstm_autoencoder = Sequential()
# Encoder
inputs = keras.Input(shape=(input_dim1, input_dim2))
lstm_autoencoder.add(LSTM(32, activation='relu', input_shape=(input_dim1, input_dim2), return_sequences=True))
lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=False))
lstm_autoencoder.add(RepeatVector(timesteps))
# Decoder
lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=True))
lstm_autoencoder.add(LSTM(32, activation='relu', return_sequences=True))
lstm_autoencoder.add(TimeDistributed(Dense(n_features)))

lstm_autoencoder.summary()
lstm_autoencoder.compile(loss='mae', optimizer='adam')

# ###############


autoencoder = autoencoder_model.autoencoder_lstm(input_dim1, input_dim2)
autoencoder.compile(**param["fit_lstm"]["compile"])
history = autoencoder.fit(
            x_train,
            y_train,
            epochs=param["fit_lstm"]["epochs"],
            batch_size=param["fit_lstm"]["batch_size"],
            verbose=param["fit_lstm"]["verbose"],
            validation_split=param["fit_lstm"]["validation_split"],
            # validation_data=(x_test, x_test),
            shuffle=param["fit_lstm"]["shuffle"])

autoencoder.summary()

