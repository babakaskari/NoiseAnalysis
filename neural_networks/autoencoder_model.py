from tensorflow.keras.layers import Input, Dense
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Embedding


def anomaly_detector(input):
    inputlayer = Input(shape=(input,))
    print("input layer : ", inputlayer)
    print("input  : ", input)
    model_layer = Dense(64, activation="relu")(inputlayer)
    model_layer = Dense(64, activation="relu")(model_layer)
    model_layer = Dense(8, activation="relu")(model_layer)
    model_layer = Dense(64, activation="relu")(model_layer)
    model_layer = Dense(64, activation="relu")(model_layer)
    model_layer = Dense(input, activation="sigmoid")(model_layer)
    print("model_layer : ", model_layer)
    return Model(inputs=inputlayer, outputs=model_layer)


def encoder_layer(input):
    inputlayer = Input(shape=(input,))
    model_layer = Dense(64, activation="relu")(inputlayer)
    model_layer = Dense(64, activation="relu")(model_layer)
    model_layer = Dense(8, activation="relu")(model_layer)
    return Model(inputs=inputlayer, outputs=model_layer)


def decoder_layer(input):
    inputlayer = Input(shape=(input,))
    model_layer = Dense(64, activation="relu")(inputlayer)
    model_layer = Dense(64, activation="relu")(model_layer)
    model_layer = Dense(8, activation="sigmoid")(model_layer)
    print("model_layer : ", model_layer)
    return Model(inputs=inputlayer, outputs=model_layer)


class AnomalyDetector(Model):
    def __init__(self, input_dim):
        super(AnomalyDetector, self).__init__()
        self.encoder = encoder_layer(input_dim)
        self.decoder = decoder_layer(input_dim)

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AnomalyDetectorTest(Model):
    def __init__(self, input_dim):
        super(AnomalyDetectorTest, self).__init__()
        self.encoder = tf.keras.Sequential([
          layers.Dense(8, activation="relu", input_shape=(input_dim,)),
          layers.Dense(6, activation="relu"),
          layers.Dense(6, activation="relu"),
          layers.Dense(4, activation="relu"),
          layers.Dense(2, activation="relu")])
        self.decoder = tf.keras.Sequential([
          layers.Dense(4, activation="relu"),
          layers.Dense(6, activation="relu"),
          layers.Dense(6, activation="relu"),
          layers.Dense(8, activation="relu"),
          layers.Dense(8, activation="sigmoid")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

