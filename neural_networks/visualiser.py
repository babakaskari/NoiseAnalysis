import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import OneHotEncoder
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import preprocessing, metrics
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, max_error
from sklearn.metrics import confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import yaml
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import warnings
import seaborn as sns
sns.set()

warnings.filterwarnings('ignore')

# def label_write(plt, x_axis, y_axis):
#         for x,y in zip(x_axis, y_axis):
#             label = "{:.2f}".format(y)
#             plt.annotate(label, # this is the text
#             (x,y), # this is the point to label
#             textcoords="offset points", # how to position the text
#             xytext=(0,10), # distance from text to points (x,y)
#             ha='center') # horizontal alignment can be left, right or center


def train_val_loss_plotter(history):
    with open("initializer.yaml") as stream:
        param = yaml.safe_load(stream)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(param["fit"]["epochs"])
    plt.figure()
    plt.plot(epochs, loss, 'b', c='red', label='Training loss')
    plt.plot(epochs, val_loss, 'b', c='blue', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def precision_recall_plotter(x_valid, valid_prediction, df_valid, history):
    # mse = np.mean(np.power(df_valid_x_rescaled - valid_x_predictions, 2), axis=1)

    mse = np.mean(np.power(x_valid - valid_prediction, 2), axis=1)

    error_df = pd.DataFrame({'Reconstruction_error': mse,
                             'True_class': df_valid})
    # print("error_def : ", error_df)
    precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
    print("Precision : ", precision_rt)
    print("Recall : ", recall_rt)
    plt.plot(threshold_rt, precision_rt[1:], label="Precision", linewidth=5)
    plt.plot(threshold_rt, recall_rt[1:], label="Recall", linewidth=5)
    plt.title('Precision and recall for different threshold values')
    plt.xlabel('Threshold')
    plt.ylabel('Precision/Recall')
    plt.legend()
    plt.show()