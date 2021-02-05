import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm


def GridSearchCVFunc(kernel, x_train, y_train, param):
    g_search = GridSearchCV(estimator=kernel,
                        param_grid=param,
                        cv=3,
                        n_jobs = -1,
                        verbose = 4)
    g_search.fit(x_train, y_train.values.ravel())
    # print("model hypermarateres : ", model.get_params())
    print("Best hyperparameters : ", g_search.best_params_)
    # print(g_search.best_params_["criterion"])
    return g_search.best_params_


def rfr_hyperparameter_tuner(model, x_train, y_train):
    RandomForestRegressor_parameters = {
        "n_estimators": np.arange(100, 1000, 100),
        "criterion": ['mse', 'mae'],
        "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        "min_samples_split": [2, 4, 6, 8, 10],                                      
        "min_samples_leaf": [1, 2, 4, 6, 8, 10],
        "max_features": ['auto', 'sqrt'],
        "bootstrap": [True, False],
                    }
    return GridSearchCVFunc(model, x_train, y_train, RandomForestRegressor_parameters)

    
def svm_hyperparameter_tuner(model, x_train, y_train):    
    svm_parameters = {           
           "kernel": ["linear", "rbf", "poly"],
           "degree": [2, 3, 4, 5],
           "gamma": ["scale", "auto"], 
           "coef0": [0.0, 0.5, 1.0, 1.5, 2.0],
           "epsilon": [0.05, 0.1, 0.2, 0.3, 0.5],
        
    }
    return GridSearchCVFunc(model, x_train, y_train, svm_parameters)


def knn_hyperparameter_tuner(model, x_train, y_train):
    knn_parameters = {    
        "algorithm": ["kd_tree", "auto", "ball_tree", "brute"], 
        "leaf_size": np.arange(10, 30, 10),     
        "n_neighbors": np.arange(2, 4, 5),
        "n_jobs": [-1],       
        }
    return GridSearchCVFunc(model, x_train, y_train, knn_parameters)


def ridge_hyperparameter_tuner(model, x_train, y_train):
    ridge_parameters = {    
        "alpha": [0.05, 0.1, 0.2, 0.3, 0.5], 
        "fit_intercept": [True, False],     
        "normalize": [True, False],
        "max_iter": [2000],       
        }
    return GridSearchCVFunc(model, x_train, y_train, ridge_parameters)


def xgb_hyperparameter_tuner(model, x_train, y_train):
    xgb_parameters = {    
        "n_estimators": np.arange(10, 30, 10),
        'max_depth': np.arange(3, 15, 1),
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7],
        'booster': ["gbtree", "gblinear", "dart"],
        'eval_metric': ["map","error","auc"],
        'subsample': [0.9, 1.0],
        'colsample_bytree': [0.9, 1.0], 
        'n_jobs': [-1],  
        'verbosity': [0],
        }
    return GridSearchCVFunc(model, x_train, y_train, xgb_parameters)


def adaboost_hyperparameter_tuner(model, x_train, y_train):
    ada_parameters = {    
        "n_estimators": np.arange(10, 30, 10),         
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7],
        'loss' : ['linear', 'square', 'exponential'],
        }
    return GridSearchCVFunc(model, x_train, y_train, ada_parameters)


def et_hyperparameter_tuner(model, x_train, y_train):
    ExtraTreesr_parameters = {
        "n_estimators": np.arange(100, 1000, 100),
        "criterion": ['mse', 'mae'],
        "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],  
        "max_features": ['auto', 'sqrt'],
        "bootstrap": [True, False],
        "n_jobs": [-1],
        }
    return GridSearchCVFunc(model, x_train, y_train, ExtraTreesr_parameters)