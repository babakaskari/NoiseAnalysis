import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import preprocessing, metrics
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, max_error
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# tf.keras.losses.MeanSquaredError(
#     reduction=losses_utils.ReductionV2.AUTO, name='mean_squared_error'
# )
import warnings

# import data_picker
import seaborn as sns
sns.set()

warnings.filterwarnings('ignore')


def evaluate_ann(history):
    # print("Cosine Proximity  :   ", history.history['cosine'])
    print("Train Loss : ", history.history["loss"])
    print("Validation Loss : ", history.history["val_loss"])
    print("Mean Squared Erro : ", history.history['mean_squared_error'])
    print("Mean Absolute Error : ", history.history['mean_absolute_error'])
    print("Mean Absolute Percentage Error : ", history.history['mean_absolute_percentage_error'])
    print("Val Mean Squared Erro : ", history.history['val_mean_squared_error'])
    print("Val Mean Absolute Error : ", history.history['val_mean_absolute_error'])
    print("Val Mean Absolute Percentage Error : ", history.history['val_mean_absolute_percentage_error'])
    # plot metrics

 






