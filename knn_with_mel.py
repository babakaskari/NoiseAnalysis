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
import evaluator

# sys.path.insert(0, "C:\\Graphviz\\bin")
# sys.path.insert(0, "C:\\Graphviz")
# sns.set()

file_to_read = open('.\\pickle\\preprocessed_dataset_with_mel.pickle', "rb")
loaded_object = pickle.load(file_to_read)
file_to_read.close()
# print(loaded_object)
dataset = loaded_object
x_dataset = dataset["x_dataset"]
y_dataset = dataset["y_dataset"]
n_x_dataset = dataset["n_x_dataset"]

# print("dataset : ", dataset)

print("x_dataset : \n", x_dataset.shape)
print("y_dataset : \n", y_dataset.shape)
print("normalized x_dataset : \n", n_x_dataset.shape)
x_train, x_test, y_train, y_test = train_test_split(x_dataset,
                                                    y_dataset,
                                                    test_size=0.30,
                                                    shuffle=True,
                                                    stratify=y_dataset,
                                                    random_state=42)
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(x_train, y_train)
print("x_train : ", x_train)
print("x_test : ", x_test)
print("y_train : ", y_train)
print("y_test : ", y_test)
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

