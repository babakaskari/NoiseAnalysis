import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import plot_tree
import pickle
import time
from matplotlib import dates as mpl_dates
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import sys
import evaluator
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
print("result shape : ", result.shape)

# result = dataset["result"]
print("result : ", result)
print("result shpae : ", result.shape)

# clf = XGBClassifier()
y_dataset = result.loc[:, ["label"]]
x_dataset = result.drop(["label"], axis=1)
# print("x_dataset : ", x_dataset)
# print("y_dataset : ", y_dataset)
x_train, x_test, y_train, y_test = train_test_split(x_dataset,
                                                    y_dataset,
                                                    test_size=0.25,
                                                    shuffle=True,
                                                    stratify=y_dataset,
                                                    random_state=42)
print("x_train : ", x_train)
print("x_test : ", x_test)
clf = xgb.sklearn.XGBClassifier(n_estimators=100,
                                max_depth=10,
                                learning_rate=0.3,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                booster="gbtree",
                                eval_metric="error",
                                scale_pos_weight=5,
                                seed=42)
clf.fit(x_train, y_train)
# xgb_pred = clf.predict(x_test)

evaluator.evaluate_preds(clf, x_train, y_train, x_test, y_test)

# plt.figure(figsize=(20, 15))
# xgb.plot_importance(clf, ax=plt.gca())
# plt.show()
# plt.figure(figsize=(20, 15))
# xgb.plot_tree(clf, ax=plt.gca())
# plt.show()
# print("Number of boosting trees: {}".format(clf.n_estimators))
# print("Max depth of trees: {}".format(clf.max_depth))
# print("Objective function: {}".format(clf.objective))
# plot_tree(clf, num_trees=10)
plt.show()

