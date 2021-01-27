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


def preprocessed_mimi():

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
    print("train_files  : \n", train_files)
    print("train  labels : \n", y_train)
    print("test files : \n", test_files)
    print("test labels : \n", y_test)
    print("train_files  shape :  ", len(train_files))
    print("train  labels shape :  ", y_train.size)
    print("test files shape :  ", test_files.shape)
    print("test labels  shape:   ", y_test.shape)

    for idx in range(len(train_files)):
        try:
            multi_channel_data, sr = librosa.load(train_files[idx], sr=None, mono=False)
            print("idx : ", idx)
            print("name of the file : ", train_files[idx])
            print("multi channel data size  : ", multi_channel_data.size)
            print("multi channel data shape  : ", multi_channel_data.shape)
            print("sr   : ", sr)
            print("multi channel data  : ", multi_channel_data)
            # plt.plot(multi_channel_data)
            # plt.title('Signal')
            # plt.xlabel('Time (samples)')
            # plt.ylabel('Amplitude')
            # plt.show()
            if multi_channel_data.ndim <= 1:
                sr = sr
                y = multi_channel_data

            sr = sr
            y = numpy.array(multi_channel_data)[0, :]

        except ValueError as msg:
            logger.warning(f'{msg}')

        print("y before mel : ", y)
        print("y size : ", y.size)
        print("y shpae : ", y.shape)
        mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                         sr=sr,
                                                         n_fft=n_fft,
                                                         hop_length=hop_length,
                                                         n_mels=n_mels,
                                                         power=power)
        print("mel spectrogram : ", mel_spectrogram)
        print("mel spectrogram shape : ", mel_spectrogram.shape)
        # 03 convert melspectrogram to log mel energy
        log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)

        print("log mel spectrogram : ", log_mel_spectrogram)
        # 04 calculate total vector size
        vectorarray_size = len(log_mel_spectrogram[0, :]) - frames + 1
        print("vector size : ", vectorarray_size)
        # 05 skip too short clips
        if vectorarray_size < 1:
            vectorarray = numpy.empty((0, dims), float)

        # 06 generate feature vectors by concatenating multi_frames
        vectorarray = numpy.zeros((vectorarray_size, dims), float)

        for t in range(frames):
            vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T

        if idx == 0:
            x_train = numpy.zeros((vectorarray.shape[0] * len(train_files), dims), float)

        x_train[vectorarray.shape[0] * idx: vectorarray.shape[0] * (idx + 1), :] = vectorarray

    print("x_train : ", x_train)

    for idx in range(len(test_files)):
        try:
            multi_channel_data, sr = librosa.load(test_files[idx], sr=None, mono=False)
            if multi_channel_data.ndim <= 1:
                sr = sr
                y = multi_channel_data

            sr = sr
            y = numpy.array(multi_channel_data)[0, :]

        except ValueError as msg:
            logger.warning(f'{msg}')

        mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                         sr=sr,
                                                         n_fft=n_fft,
                                                         hop_length=hop_length,
                                                         n_mels=n_mels,
                                                         power=power)

        # 03 convert melspectrogram to log mel energy
        log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)

        # 04 calculate total vector size
        vectorarray_size = len(log_mel_spectrogram[0, :]) - frames + 1
        # 05 skip too short clips
        if vectorarray_size < 1:
            vectorarray = numpy.empty((0, dims), float)

        # 06 generate feature vectors by concatenating multi_frames
        vectorarray = numpy.zeros((vectorarray_size, dims), float)

        for t in range(frames):
            vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T

        if idx == 0:
            x_test = numpy.zeros((vectorarray.shape[0] * len(test_files), dims), float)

        x_test[vectorarray.shape[0] * idx: vectorarray.shape[0] * (idx + 1), :] = vectorarray

    print("x_test : ", x_test)
    x_train = pd.DataFrame(x_train)
    # print("x_train : ", x_train)
    # print("description : \n", x_train.describe)


