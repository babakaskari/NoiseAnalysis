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
import neural_network_evaluator
import visualiser
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
datatset = dataset["n_result"]

df_train, df_test = train_test_split(datatset,
                                     test_size=param["data_split"]["test_size"],
                                     shuffle=param["data_split"]["shuffle"],
                                     random_state=param["data_split"]["random_state"])

df_train, df_valid = train_test_split(df_train,
                                      test_size=param["data_split"]["test_size"],
                                      random_state=param["data_split"]["random_state"])
# print('x_train : ', df_train)
train_normal = df_train.loc[df_train['label'] == 0]
train_abnormal = df_train.loc[df_train['label'] == 1]
x_train = df_train.drop(['label'], axis=1)
x_train_normal = train_normal.drop(['label'], axis=1)
x_train_abnormal = train_abnormal.drop(['label'], axis=1)

test_normal = df_test.loc[df_test['label'] == 0]
test_abnoram = df_test.loc[df_test['label'] == 1]
x_test = df_test.drop(['label'], axis=1)
x_test_normal = test_normal.drop(['label'], axis=1)
x_test_abnoral = test_abnoram.drop(['label'], axis=1)

valid_normal = df_valid.loc[df_valid['label'] == 0]
valid_abnormal = df_valid.loc[df_valid['label'] == 1]
x_valid = df_valid.drop(['label'], axis=1)
x_valid_normal = valid_normal.drop(['label'], axis=1)
x_valid_abnormal = valid_abnormal.drop(['label'], axis=1)

input_dim = x_train_normal.shape[1]

print("input_dim :", input_dim)
# print("x_train : ", x_train_normal)
# print("x_valid : ", x_valid_normal)
# autoencoder = autoencoder_model.autoencoder_anomaly_detection(input_dim)
autoencoder = autoencoder_model.AnomalyDetectorAutoencoder(input_dim)

# autoencoder.summary()
# Model summary
autoencoder.compile(**param["fit"]["compile"])

# cp = ModelCheckpoint(filepath="autoencoder_classifier.h5",
#                      save_best_only=True,
#                      verbose=0)
cp = tf.keras.callbacks.ModelCheckpoint(
             filepath="autoencoder_classifier.h5",
             save_weights_only=True,
             monitor='val_accuracy',
             mode='max',
             save_best_only=True)
tb = tf.keras.callbacks.TensorBoard(log_dir='.\\logs',
                                    histogram_freq=0,
                                    write_graph=True,
                                    write_images=True)
x_train_normal_value = x_train_normal.values
# print("x_train_normal_value : ", x_train_normal_value)

history = autoencoder.fit(
            x_train_normal_value,
            x_train_normal_value,
            epochs=param["fit"]["epochs"],
            batch_size=param["fit"]["batch_size"],
            verbose=param["fit"]["verbose"],
            # validation_split=param["fit"]["validation_split"],
            validation_data=(x_valid_normal, x_valid_normal),
            shuffle=param["fit"]["shuffle"],
            callbacks=[cp, tb])

autoencoder.encoder.summary()
autoencoder.decoder.summary()
autoencoder.summary()
neural_network_evaluator.evaluate_ann(history)
visualiser.train_val_loss_plotter(history)

x_valid_predictions = autoencoder.predict(x_valid)
visualiser.precision_recall_plotter(x_valid, x_valid_predictions, df_valid["label"], history)

test_x_predictions = autoencoder.predict(df_test_x_rescaled)
mse = np.mean(np.power(df_test_x_rescaled - test_x_predictions, 2), axis=1)
error_df_test = pd.DataFrame({'Reconstruction_error': mse,
                        'True_class': df_test['y']})
error_df_test = error_df_test.reset_index()
threshold_fixed = 0.4
groups = error_df_test.groupby('True_class')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Break" if name == 1 else "Normal")
ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show();