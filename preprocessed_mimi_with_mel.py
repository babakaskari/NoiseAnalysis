#!/usr/bin/env python
"""
 @file   baseline.py
 @brief  Baseline code of simple AE-based anomaly detection used experiment in [1].
 @author Ryo Tanabe and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2019 Hitachi, Ltd. All right reserved.
 [1] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection," arXiv preprint arXiv:1909.09347, 2019.
"""
########################################################################
# import default python-library
########################################################################
import pickle
import os
import sys
import glob
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy as np
import librosa
import librosa.core
import librosa.feature
import yaml
import logging
from sklearn.preprocessing import StandardScaler
# from import
from tqdm import tqdm
from sklearn import metrics
from keras.models import Model
from keras.layers import Input, Dense
########################################################################


########################################################################
# version
########################################################################
__versions__ = "1.0.3"
########################################################################


########################################################################
# setup STD I/O
########################################################################
"""
Standard output is logged in "baseline.log".
"""
logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
########################################################################


########################################################################
# visualizer
########################################################################
class visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(30, 10))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):

        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Test"], loc="upper right")

    def save_figure(self, name):

        self.plt.savefig(name)


########################################################################


########################################################################
# file I/O
########################################################################
# pickle I/O
def save_pickle(filename, save_data):

    logger.info("save_pickle -> {}".format(filename))
    with open(filename, 'wb') as sf:
        pickle.dump(save_data, sf)


def load_pickle(filename):

    logger.info("load_pickle <- {}".format(filename))
    with open(filename, 'rb') as lf:
        load_data = pickle.load(lf)
    return load_data


# wav file Input
def file_load(wav_name, mono=False):

    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))


def demux_wav(wav_name, channel=0):

    try:
        multi_channel_data, sr = file_load(wav_name)
        if multi_channel_data.ndim <= 1:
            return sr, multi_channel_data

        return sr, np.array(multi_channel_data)[channel, :]

    except ValueError as msg:
        logger.warning(f'{msg}')


########################################################################


########################################################################
# feature extractor
########################################################################
def file_to_vector_array(file_name,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):

    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa (**kwargs == param["librosa"])
    sr, y = demux_wav(file_name)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)

    # 04 calculate total vector size
    vectorarray_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vectorarray_size < 1:
        return np.empty((0, dims), float)

    # 06 generate feature vectors by concatenating multi_frames
    vectorarray = np.zeros((vectorarray_size, dims), float)
    for t in range(frames):
        vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T

    return vectorarray


def list_to_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):

    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 loop of file_to_vectorarray
    for idx in tqdm(range(len(file_list)), desc=msg):

        vector_array = file_to_vector_array(file_list[idx],
                                            n_mels=n_mels,
                                            frames=frames,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            power=power)

        if idx == 0:
            dataset = np.zeros((vector_array.shape[0] * len(file_list), dims), float)

        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array

    return dataset


def dataset_generator():



    # 01 normal list generate
    normal_files = sorted(glob.glob(".\\data\\normal\\*.wav"))

    normal_labels = np.zeros(len(normal_files))
    print("normal files : ", normal_files)
    if len(normal_files) == 0:
        logger.exception("no_wav_data!!")

    # 02 abnormal list generate
    abnormal_files = sorted(glob.glob(".\\data\\abnormal\\*.wav"))
    abnormal_labels = np.ones(len(abnormal_files))
    print("abnormal files : ", abnormal_files)
    if len(abnormal_files) == 0:
        logger.exception("no_wav_data!!")

    # 03 separate train & eval
    train_files = normal_files[:]
    print("train files : ", train_files)
    print("train files length: ", len(train_files))
    train_labels = normal_labels[:]
    print("train labels length: ", len(train_labels))
    eval_files = abnormal_files[:]
    print("eval files : ", eval_files)
    eval_labels = abnormal_labels[:]
    print("eval labels : ", eval_labels)
    logger.info("train_file num : {num}".format(num=len(train_files)))
    logger.info("eval_file  num : {num}".format(num=len(eval_files)))

    return train_files, train_labels, eval_files, eval_labels
# main
########################################################################

if __name__ == "__main__":

    # load parameter yaml
    with open("baseline.yaml") as stream:
        param = yaml.safe_load(stream)

    # make output directory
    # initialize the visualizer
    visualizer = visualizer()

    # setup the result
    result_file = "{result}/{file_name}".format(result=param["result_directory"], file_name=param["result_file"])
    results = {}

    # loop of the base directory

    print("\n===========================")

    # setup path

    train_files, train_labels, eval_files, eval_labels = dataset_generator()
    print("train files: ", train_files)
    print("eval files : ", eval_files)

    train_data = list_to_vector_array(train_files,
                                      msg="generate train_dataset",
                                      n_mels=param["feature"]["n_mels"],
                                      frames=param["feature"]["frames"],
                                      n_fft=param["feature"]["n_fft"],
                                      hop_length=param["feature"]["hop_length"],
                                      power=param["feature"]["power"])

    eval_data = list_to_vector_array(eval_files,
                                      msg="generate train_dataset",
                                      n_mels=param["feature"]["n_mels"],
                                      frames=param["feature"]["frames"],
                                      n_fft=param["feature"]["n_fft"],
                                      hop_length=param["feature"]["hop_length"],
                                      power=param["feature"]["power"])

    # # model training
    # print("============== MODEL TRAINING ==============")
    # model = keras_model(param["feature"]["n_mels"] * param["feature"]["frames"])
    # model.summary()
    # training
    print("train files   : ", train_files)
    print("train_data   : ", train_data)
    print("train_data  shape : ", train_data.shape)
    print("train labels   : ", train_labels)
    print("eval_files : ", eval_files)
    print("evale labels : ", eval_labels)
    print("evale data : ", eval_data)
    x_dataset = np.concatenate((train_data, eval_data), axis=0)
    scaler = StandardScaler()
    # # print(y_dataset)
    n_x_dataset = scaler.fit_transform(x_dataset)
    print("final x_dataset  :   \n", x_dataset)
    print("ohe final x_dataset  :   \n", n_x_dataset)
    y_dataset = np.concatenate((train_labels, eval_labels), axis=0)
    print("final y_dataset  :   \n", y_dataset)

    data_dict = {
        "x_dataset": x_dataset,
        "y_dataset": y_dataset,
        "n_x_dataset": n_x_dataset,

    }

    f_t_write = open('.\\pickle\\preprocessed_dataset_with_mel.pickle', "wb")
    pickle.dump(data_dict, f_t_write)
    f_t_write.close()
########################################################################
