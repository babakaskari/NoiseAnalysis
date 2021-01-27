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
from keras.models import Model
from keras.layers import Input, Dense
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

normal_labels = numpy.zeros(len(normal_files))
if len(normal_files) == 0:
    logger.exception("no_wav_data!!")

# 02 abnormal list generate
abnormal_files = sorted(glob.glob(".\\data\\abnormal\\*.wav"))

abnormal_labels = numpy.ones(len(abnormal_files))

if len(abnormal_files) == 0:
    logger.exception("no_wav_data!!")

train_files = normal_files[len(abnormal_files):]
y_train = normal_labels[len(abnormal_files):]
test_files = numpy.concatenate((normal_files[:len(abnormal_files)], abnormal_files), axis=0)
y_test = numpy.concatenate((normal_labels[:len(abnormal_files)], abnormal_labels), axis=0)
# print("train_files  : \n", train_files)
print("train  labels shape: \n", y_train.shape)
i = 0
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
            # print(i)
            df1 = pd.DataFrame(multi_channel_data.reshape(1, -1))
            df_test = df_test.append(df1)

        else:
            # print(i)
            df_test = pd.DataFrame(data=multi_channel_data.reshape(1, -1))
            i = i + 1

    except ValueError as msg:
        logger.warning(f'{msg}')
# df_test = df_test.abs()
x_test = df_test.reset_index()
# print("x_test : ", x_test)
# print("y_test : ", y_test)
# print("y_test shape: ", y_test.shape)
# print("x_train maximum : ", x_train.max(axis=1))
# print("x_test maximum : ", x_test.max(axis=1))
# print("x_train minimum : ", x_train.min(axis=1))
# print("x_test minimum : ", x_test.min(axis=1))
# print("x_train maximum : ", x_train.max(axis=1))
x_train_description = x_train.apply(pd.DataFrame.describe, axis=1)
x_test_description = x_test.apply(pd.DataFrame.describe, axis=1)
# print("x_train description : ", x_train_description)
# print("x_test description : ", x_test_description)
# print("x_train median : ", x_train.median(axis=1))
# print("x_test median : ", x_test.median(axis=1))
# x_train.to_csv("x_train.csv", index=False)
# x_test.to_csv("x_test.csv", index=False)
x_train_median = pd.DataFrame(x_train.median(axis=1), columns=['median'])
x_test_median = pd.DataFrame(x_test.median(axis=1), columns=['median'])
x_train_description = pd.concat([x_train_description, x_train_median], axis=1)
x_test_description = pd.concat([pd.DataFrame(x_train), x_test_median], axis=1)
x_train_description.to_csv(".\\result\\x_train_description.csv", index=True)
x_test_description.to_csv(".\\result\\x_test_description.csv", index=True)
# print("x_train : ", x_train)
# print("x_test : ", x_test)
x_result = pd.concat([pd.DataFrame(x_train), pd.DataFrame(x_test)], axis=0)
y_result = pd.concat([pd.DataFrame(y_train), pd.DataFrame(y_test)], axis=0)
y_result.columns = ['label']
result = pd.concat([x_result, y_result], axis=1)
result.drop(["index"], axis=1, inplace=True)
# print("result : ", result)
datat_dict = {
    # "x_train": x_train,
    # "y_train": y_train,
    # "x_test": x_test,
    # "y_test": y_test,
    "result": result,
}

f_t_write = open('.\\pickle\\preprocessed_dataset.pickle', "wb")
pickle.dump(datat_dict, f_t_write)
f_t_write.close()
# return datat_dict


