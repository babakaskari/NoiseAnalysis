from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
import tensorflow as tf

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
# print("dataset : ", dataset)
# print("dataset shape : ", dataset.shape)
# dataset = preprocessed_mimi.preprocessing_mimi()
# x_train = dataset["x_train"]
# y_train = dataset["y_train"]
# x_test = dataset["x_test"]
# y_test = dataset["y_test"]

with open("initializer.yaml") as stream:
    param = yaml.safe_load(stream)
datatset = dataset["n_result"]
datatset = datatset.drop(['quantile1', 'quantile2', 'quantile3'], axis=1)
# print("datatset : \n", datatset)

# y_dataset = datatset.loc[:, ['label']]
# x_dataset = datatset.drop(['label'], axis=1)
datatset = datatset.values
# print("dataset : ", dataset)
y_dataset = datatset[:, -1]
x_dataset = datatset[:, 0:-1]
# print("x_dataset :", x_dataset)
# print("y_dataset :", y_dataset)
x_train, x_test, y_train, y_test = train_test_split(
                                                    x_dataset,
                                                    y_dataset,
                                                    test_size=param["data_split"]["test_size"],
                                                    shuffle=param["data_split"]["shuffle"],
                                                    stratify=y_dataset,
                                                    random_state=param["data_split"]["random_state"])

# print("x_test : ", x_test)
# print("y_test : ", y_test)
y_train = y_train.astype(bool)
y_test = y_test.astype(bool)
# print("y_train", y_train)
# print("y_test", y_test)
normal_train_data = x_train[~y_train]
normal_test_data = x_test[~y_test]

abnormal_train_data = x_train[y_train]
abnormal_test_data = x_test[y_test]
# print("normal_train_data", normal_train_data)
# print("abnormal_train_data", abnormal_train_data)
input_dim = x_train.shape[1]
visualiser.wave_plotter(normal_train_data, "A Normal Machine", input_dim)
visualiser.wave_plotter(abnormal_train_data, "An Abnormal Machine", input_dim)


# autoencoder = autoencoder_model.AnomalyDetectorAutoencoder(input_dim)
autoencoder = autoencoder_model.AnomalyDetectorAutoencoder_No_Q(input_dim)

# autoencoder = autoencoder_model.anomaly_detector(x_train.shape[1], )
# print("x_train input train shape: ", x_train.shape[1])
# print("output_shape  :   ", autoencoder.output_shape)
autoencoder.compile(**param["fit"]["compile"])
history = autoencoder.fit(
            normal_train_data,
            normal_train_data,
            epochs=param["fit"]["epochs"],
            batch_size=param["fit"]["batch_size"],
            verbose=param["fit"]["verbose"],
            # validation_split=param["fit"]["validation_split"],
            validation_data=(x_test, x_test),
            shuffle=param["fit"]["shuffle"])
# Model summary
autoencoder.encoder.summary()
autoencoder.decoder.summary()
autoencoder.summary()


# ===================================plotting the model as a graph start
# keras.utils.plot_model(model, "my_first_model.png")
# keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
# ===================================plotting the model as a graph end
# Model config
# print("get_config  :   ",model.get_config())

# List all weight tensors
# print("get_weights  :   ", model.get_weights())

neural_network_evaluator.evaluate_ann(history)
visualiser.train_val_loss_plotter(history)
encoded_imgs = autoencoder.encoder(normal_test_data).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
visualiser.reconstruction_error(normal_test_data, decoded_imgs, input_dim)
encoded_imgs = autoencoder.encoder(abnormal_test_data).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
visualiser.reconstruction_error(abnormal_test_data, decoded_imgs, input_dim)

reconstructions = autoencoder.predict(normal_train_data)
train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)
visualiser.histogram_plotter(train_loss, "Train loss", "No of examples")
# threshold = np.mean(train_loss) + np.std(train_loss)
# print(" threshold : ", np.mean(train_loss) + np.std(train_loss))
reconstructions = autoencoder.predict(abnormal_test_data)
test_loss = tf.keras.losses.mae(reconstructions, abnormal_test_data)
visualiser.histogram_plotter(test_loss, "Test loss", "No of examples")
threshold = param["threshold"]
print("Threshold: ", threshold)
neural_network_evaluator.prediction(autoencoder, x_test, y_test, threshold)
