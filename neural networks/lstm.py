import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
import dataset
import neural_network_evaluator
import visualiser
# import data_picker
dId = 57975963
year = 2019
month = 6
day = 21
hour = 13



computation_range = np.arange(1,5, 1)
what_hour = np.arange(1,3, 1)
dataset = dataset.main()

# x_predict = x_predict.values.ravel()
# print("x_predict shape :", x_predict.shape)
# ========================= filtering the dataset accoring to id, year, month, day, hour
# print("dataset : ", dataset)
dId_list = dataset.DeviceId.unique()
# print("dId_list : ", dId_list)

# ===============================
# print("dataset : ", dataset)
y_dataset = dataset.Value
x_dataset = dataset.drop(["Value"], axis=1)

# =============
indexH = dataset[(dataset["hour"] == hour) & (dataset["Day"] == day) & (dataset["Month"] == month)].index
print("index :: ", indexH)
print("index :: ", indexH[0])
x_predict = x_dataset.iloc[indexH[0]]
print("x_predict shape :", x_predict.shape)

x_predict = x_predict.ravel()
print("x_predict shape :", x_predict.shape)

x_predict = x_predict.reshape(-1, 1)
print("x_predict shape :", x_predict.shape)

x_predict = x_predict.reshape(1, -1)
print("x_predict shape :", x_predict.shape)
# =============

# print("y_dataset : ",y_dataset)
# print("x_dataset : ", x_dataset)
x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, shuffle=False, test_size=0.2, random_state=42)

x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, shuffle=False, test_size=0.2, random_state=42)
print("x_train : ", x_train)
print("y_train : ", y_train)
print("x_train shape : ", x_train.shape)
print("y_train shape  : ", y_train.shape)

model = Sequential()
model.add(Dense(12, activation='relu', input_shape=(7,)))
model.add(Dense(6, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# model training
# Model output shape
print("output_shape  :   ", model.output_shape)

# Model summary
model.summary()

# Model config
# print("get_config  :   ",model.get_config())

# List all weight tensors 
# print("get_weights  :   ", model.get_weights())

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae', 'mape', 'cosine'])
                   
history = model.fit(x_train, y_train,epochs=50, batch_size=1, verbose=1)
neural_network_evaluator.evaluate_ann(history, model, x_train, y_train, x_test, y_test, x_cv, y_cv, x_predict)
# history = model.fit(X, X, epochs=500, batch_size=len(X), verbose=2)
