from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
import tensorflow as tf
import yaml
import autoencoder_model
import neural_network_evaluator
import visualiser
# import neural_network_evaluator
# import visualiser

import pickle
import seaborn as sns
from sklearn import metrics
sns.set()

file_to_read = open('..\\pickle\\preprocessed_dataset_ann.pickle', "rb")
# file_to_read = open('/home/mohammed/pickle/preprocessed_dataset.pickle', "rb")
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
normal_data= dataset["normal_data"]
abnormal_data = dataset["abnormal_data"]
print("normal_data : \n", normal_data)
print("abnormal_data : \n", abnormal_data)
print("normal_data shapae : \n", normal_data.shape)
y_normal = normal_data.loc[:, ['label']]
x_normal = normal_data.drop(['label'], axis=1)
y_abnormal = abnormal_data.loc[:, ['label']]
x_abnormal = abnormal_data.drop(['label'], axis=1)

# print("x_dataset : ", x_dataset)
# print("y_dataset : ", y_dataset)
x_train, x_test, y_train, y_test = train_test_split(normal_data,
                                                    y_normal,
                                                    test_size=0.25,
                                                    shuffle=True,
                                                    stratify=y_normal,
                                                    random_state=42)

autoencoder = autoencoder_model.anomaly_detector(x_train.shape[1], )
print("x_train input train shape: ", x_train.shape[1])
print("output_shape  :   ", autoencoder.output_shape)

# Model summary
autoencoder.summary()

# ===================================plotting the model as a graph start
# keras.utils.plot_model(model, "my_first_model.png")
# keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
# ===================================plotting the model as a graph end
# Model config
# print("get_config  :   ",model.get_config())

# List all weight tensors
# print("get_weights  :   ", model.get_weights())

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
# model.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae', 'mape', 'cosine'])
# autoencoder.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae', 'mape'])
autoencoder.compile(**param["fit"]["compile"])
# autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
# history = model.fit(x_train, y_train,epochs=5, batch_size=50, verbose=1)
# //////////////////////////////////////
history = autoencoder.fit(x_train,
                          x_train,
                          epochs=param["fit"]["epochs"],
                          shuffle=param["fit"]["shuffle"],
                          batch_size=param["fit"]["batch_size"],
                          verbose=param["fit"]["verbose"],
                          # validation_data=(x_test, x_test),
                          validation_split=param["fit"]["validation_split"],
                          )

neural_network_evaluator.evaluate_ann(history)
visualiser.train_val_loss_plotter(history)



