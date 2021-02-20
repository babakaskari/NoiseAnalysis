import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import dates as mpl_dates
import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding

import seaborn as sns
from sklearn import metrics

sns.set()

# dId = 57975963
# year = 2019
# month = 6
# day = 21
# hour = 13
dId = 57975970
year = 2019
month = 6
day = 20
hour = 1
computation_range = np.arange(6, 11, 1)
# computation_range = [35]
what_hour = np.arange(3, 3, 1)

# ========================= filtering the dataset accoring to id, year, month, day, hour and weekend

def keras_model(input):
    inputs = keras.Input(shape=(input, 1))
    model = layers.LSTM(12, return_sequences=True)(inputs)
    model = layers.LSTM(12)(model)
    model = layers.Dense(10)(model)
    outputs = layers.Dense(1)(model)
    model = keras.Model(inputs=inputs, outputs=outputs, name="water_predictor")
    return model


model = keras_model(8)

print("output_shape  :   ", model.output_shape)

# Model summary
model.summary()








