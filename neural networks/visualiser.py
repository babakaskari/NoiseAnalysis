import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import OneHotEncoder
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import preprocessing, metrics
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, max_error
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
    plt.legend()
    plt.show()