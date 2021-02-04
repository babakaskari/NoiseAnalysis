import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import plot_tree
import pickle
import time
from matplotlib import dates as mpl_dates
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
import sys
import evaluator
import hyperparameter_tuner
from sklearn.metrics import roc_curve
import seaborn as sns
import xgboost as xgb
from sklearn import metrics
from xgboost import XGBClassifier
# sys.path.insert(0, "C:\\Graphviz\\bin")
# sys.path.insert(0, "C:\\Graphviz")
# sns.set()

# file_to_read = open('.\\pickle\\preprocessed_dataset.pickle', "rb")
file_to_read = open('/home/mohammed/pickle/preprocessed_dataset.pickle', "rb")
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

result = dataset["result"]
n_result = dataset["n_result"]
print("result : \n", result)
print("n_result : \n", n_result)
print("result shape : \n", result.shape)

# clf = XGBClassifier()
y_dataset = result.loc[:, ["label"]]
x_dataset = result.drop(["label"], axis=1)
n_x_dataset = n_result.drop(["label"], axis=1)
max_min_mean_median_std = n_x_dataset.drop(['quantile1', 'quantile2', 'quantile3'], axis=1)
max_min_mean = x_dataset.drop(['quantile1', 'quantile2', 'quantile3', 'median', 'std'], axis=1)
max_min_mean_std = x_dataset.drop(['quantile1', 'quantile2', 'quantile3', 'median'], axis=1)
max_min_mean_median = x_dataset.drop(['quantile1', 'quantile2', 'quantile3', 'std'], axis=1)
mean_median_std = x_dataset.drop(['quantile1', 'quantile2', 'quantile3', 'min', 'max'], axis=1)
min_max = n_x_dataset.drop(['quantile1', 'quantile2', 'quantile3', 'median', 'std', 'mean'], axis=1)
mean_std = n_x_dataset.drop(['quantile1', 'quantile2', 'quantile3', 'median', 'min', 'max'], axis=1)
median_std = n_x_dataset.drop(['quantile1', 'quantile2', 'quantile3', 'mean', 'min', 'max'], axis=1)
min_max_median = n_x_dataset.drop(['quantile1', 'quantile2', 'quantile3', 'std', 'mean'], axis=1)

# print("x_dataset : ", x_dataset)
# print("y_dataset : ", y_dataset)
x_train, x_test, y_train, y_test = train_test_split(max_min_mean_median_std,
                                                    y_dataset,
                                                    test_size=0.25,
                                                    shuffle=True,
                                                    stratify=y_dataset,
                                                    random_state=42)
# print("x_train : ", x_train)
# print("x_test : ", x_test)

clf = KNeighborsClassifier(n_neighbors=5)
# params = hyperparameter_tuner.knn_hyperparameter_tuner(clf, x_train, y_train)
# clf.set_params(**params)
clf.fit(x_train, y_train)

evaluator.evaluate_preds(clf, x_train, y_train, x_test, y_test)


