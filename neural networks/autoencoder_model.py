from tensorflow.keras.layers import Input, Dense
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Embedding


def keras_model(input):
    inputlayer = Input(shape=(input,))
    print("input layer : ", inputlayer)
    print("input  : ", input)
    h = Dense(64, activation="relu")(inputlayer)
    h = Dense(64, activation="relu")(h)
    h = Dense(8, activation="relu")(h)
    h = Dense(64, activation="relu")(h)
    h = Dense(64, activation="relu")(h)
    h = Dense(input, activation=None)(h)

    return Model(inputs=inputlayer, outputs=h)