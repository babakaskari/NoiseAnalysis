from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
tf.enable_eager_execution()
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import auc, roc_curve
import pandas as pd
import yaml
from keras import layers
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

threshold_fixed = param["threshold_fixed"]
LABELS = param["LABEL"]
datatset = dataset["n_result"]


x_test_normal = dataset["normal_data"].iloc[0:dataset["abnormal_data"].shape[0]]
train = dataset["normal_data"].iloc[dataset["abnormal_data"].shape[0]:].reset_index(drop=True)
# print("x_test_normal : ", x_test_normal)
test = pd.concat([x_test_normal, dataset["abnormal_data"]], axis=0, ignore_index=True)
# print("train : ", train)
# print("test : ", test)
# print(train.shape, test.shape)
train, valid = train_test_split(train,
                                test_size=param["lstm_data_split"]["test_size"],
                                shuffle=param["lstm_data_split"]["shuffle"],
                                random_state=param["lstm_data_split"]["random_state"])
y_train = train["label"]
x_train = train.drop(['label'], axis=1)
y_valid = valid["label"]
x_valid = valid.drop(['label'], axis=1)
y_test = test["label"]
x_test = test.drop(['label'], axis=1)

# print(x_train.shape, x_test.shape)
x_train = x_train.to_numpy()
x_train_scaled = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_valid = x_valid.to_numpy()
x_valid_scaled = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))
x_test = x_test.to_numpy()
x_test_scaled = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# print("x_train shape :   \n", x_train.shape)
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# print(y_test)
input_dim1 = x_train_scaled.shape[1]
input_dim2 = x_train_scaled.shape[2]
print(input_dim1)
print(input_dim2)

# ######################################

autoencoder = autoencoder_model.autoencoder_model(input_dim1, input_dim2)

print("output_shape  :   ", autoencoder.output_shape)

# Model summary
autoencoder.summary()
# ######################################
autoencoder = autoencoder_model.autoencoder_lstm(input_dim1, input_dim2)
autoencoder.compile(**param["fit_lstm"]["compile"])
history = autoencoder.fit(
            x_train_scaled,
            x_train_scaled,
            epochs=param["fit_lstm"]["epochs"],
            batch_size=param["fit_lstm"]["batch_size"],
            verbose=param["fit_lstm"]["verbose"],
            validation_split=param["fit_lstm"]["validation_split"],
            # validation_data=(x_test, x_test),
            shuffle=param["fit_lstm"]["shuffle"])


neural_network_evaluator.evaluate_ann(history)
visualiser.train_val_loss_plotter(history)
# ###########################


def flatten(x):
    flattened_x = np.empty((x.shape[0], x.shape[2]))
    for i in range(x.shape[0]):
        flattened_x[i] = x[i, (x.shape[1] - 1), :]
    return flattened_x


# #########################

x_valid_predictions = autoencoder.predict(x_valid_scaled)
# mse = np.mean(np.power(flatten(x_valid) - flatten(x_valid_predictions), 2), axis=1)
mse = np.mean(np.power(flatten(x_valid_scaled) - flatten(x_valid_predictions), 2), axis=1)

# error_df = pd.DataFrame({'Reconstruction_error': mse,
#                         'True_class': x_valid['label'].tolist()})
# ###################

error_df = pd.DataFrame({'Reconstruction_error': mse,
                         'True_class': valid['label']})

# ###############################
precision, recall, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
plt.plot(threshold_rt, precision[1:], label="Precision",linewidth=5)
plt.plot(threshold_rt, recall[1:], label="Recall",linewidth=5)
plt.title('Precision and recall for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.legend()
plt.show()

# ##############################

x_test_predictions = autoencoder.predict(x_test_scaled)
mse = np.mean(np.power(flatten(x_test_scaled) - flatten(x_test_predictions), 2), axis=1)

error_df = pd.DataFrame({'Reconstruction_error': mse,
                        'True_class': test['label']})

groups = error_df.groupby('True_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Break" if name == 1 else "Normal")
ax.hlines(param["threshold_lstm"], ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show()

# ###################################################
pred_y = [1 if e > param["threshold_lstm"] else 0 for e in error_df.Reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.True_class, pred_y)

plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# ###################################
false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df.True_class, error_df.Reconstruction_error)
roc_auc = auc(false_pos_rate, true_pos_rate,)

plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f'% roc_auc)
plt.plot([0, 1], [0, 1], linewidth=5)

plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('Receiver operating characteristic curve (ROC)')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# ##################################################
print(" Precision : ", precision)
print(" Recall : ", recall)


