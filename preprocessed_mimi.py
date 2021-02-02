########################################################################
# import default python-library
########################################################################
import pickle
import os
import sys
import glob
from tqdm import tqdm
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy
import pandas as pd
import librosa
import librosa.core
import librosa.feature
import yaml
import logging
# from import
from tqdm import tqdm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import librosa.display
import matplotlib.pyplot as plt
########################################################################

n_mels = 64
frames = 5
n_fft = 1024
hop_length = 512
power = 2.0
dims = n_mels * frames

"""
Standard output is logged in "baseline.log".
"""
logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

normal_files = sorted(glob.glob(".\\data\\normal\\*.wav"))
# normal_files = sorted(glob.glob("/home/mohammed/data/normal/*.wav"))

normal_labels = numpy.zeros(len(normal_files))
if len(normal_files) == 0:
    logger.exception("no_wav_data!!")

# 02 abnormal list generate
abnormal_files = sorted(glob.glob(".\\data\\abnormal\\*.wav"))
# abnormal_files = sorted(glob.glob("/home/mohammed/data/abnormal/*.wav"))

abnormal_labels = numpy.ones(len(abnormal_files))

if len(abnormal_files) == 0:
    logger.exception("no_wav_data!!")
# print("normal files : ", normal_files)
# print("abnormal file : ", abnormal_files)
# print("normal length : ", len(normal_files))
# print("abnormal length : ", len(abnormal_files))

train_files = normal_files[:]
y_train = normal_labels[:]
# print("normal label : ", y_train)
# print("normal label shape : ", y_train.shape)
test_files = abnormal_files[:]
y_test = abnormal_labels[:]
# print("normal label : ", y_test)
# print("normal label shape : ", y_test.shape)

i = 0
df_train = pd.DataFrame()
df_test = pd.DataFrame()
for idx in range(len(train_files)):
    try:
        multi_channel_data, sr = librosa.load(train_files[idx], sr=None, mono=True)

        if i != 0:
            df1 = pd.DataFrame(multi_channel_data.reshape(1, -1))
            df_train = df_train.append(df1)

        else:
            df_train = pd.DataFrame(data=multi_channel_data.reshape(1, -1))
            i = i + 1

    except ValueError as msg:
        logger.warning(f'{msg}')
# df_train = df_train.abs()
x_train = df_train.reset_index()
# print("x_train : ", x_train)
# # print("x_train median : ", x_train.median(axis=1))
# print("y_train : ", y_train)
# print("y_train shape: ", y_train.shape)
# ////////////////////////////////////////////////
i = 0
for idx in range(len(test_files)):
    try:
        multi_channel_data, sr = librosa.load(test_files[idx], sr=None, mono=True)

        if i != 0:
            df1 = pd.DataFrame(multi_channel_data.reshape(1, -1))
            df_test = df_test.append(df1)

        else:
            df_test = pd.DataFrame(data=multi_channel_data.reshape(1, -1))
            i = i + 1

    except ValueError as msg:
        logger.warning(f'{msg}')
# df_test = df_test.abs()
x_test = df_test.reset_index()

dataset = pd.concat([pd.DataFrame(x_train), pd.DataFrame(x_test)], axis=0)

dataset.drop(["index"], axis=1, inplace=True)
# print("dataset : ", dataset)
x_dataset = pd.DataFrame()
x_dataset["min"] = dataset.min(axis=1)
x_dataset["max"] = dataset.max(axis=1)
x_dataset["mean"] = dataset.mean(axis=1)
x_dataset["median"] = dataset.median(axis=1)
x_dataset["quantile1"] = dataset.quantile(0.25)
x_dataset["quantile2"] = dataset.quantile(0.5)
x_dataset["quantile3"] = dataset.quantile(0.75)
x_dataset["std"] = dataset.std(axis=1)
x_dataset = x_dataset.reset_index()
x_dataset.drop(["index"], axis=1, inplace=True)
# print("x_dataset : ", x_dataset)
dataset_description = x_dataset.describe()
dataset_description.to_csv(".\\result\\dataset_description.csv", index=True)
# dataset_description.to_csv("/home/mohammed/result/dataset_description.csv", index=True)

# print("x_train : ", x_train)
# print("x_test : ", x_test)
y_dataset = pd.concat([pd.DataFrame(y_train), pd.DataFrame(y_test)], axis=0)
y_dataset.columns = ['label']
y_dataset = y_dataset.reset_index()
y_dataset.drop(["index"], axis=1, inplace=True)
scaler = StandardScaler()
# # print(y_dataset)
n_x_dataset = scaler.fit_transform(x_dataset)
# print("n_x_dataset : ", n_x_dataset)
y_dataset = y_dataset.applymap(int)
y_dataset = y_dataset.applymap(str)
# print("y_dataset : ", y_dataset)
result = pd.concat([x_dataset, y_dataset], axis=1)
print(" result : \n", result)
# print("x_dataset : ", x_dataset)
# print("y_dataset : ", y_dataset)
# print("result shape : ", result.shape)

data_dict = {
    "x_dataset": x_dataset,
    "y_dataset": y_dataset,
    "result": result,
}

f_t_write = open('.\\pickle\\preprocessed_dataset.pickle', "wb")
# f_t_write = open('/home/mohammed/pickle/preprocessed_dataset.pickle', "wb")
pickle.dump(data_dict, f_t_write)
f_t_write.close()
# return datat_dict


