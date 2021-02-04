import os
import glob
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from gaussrank import *
import logging
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
import numpy as np
import dataset
from sklearn.metrics import r2_score
import visualiser
import neural_network_evaluator
# import SVM_Regressor
import sys
#sys.path.insert(1, 'H:/Project/water_project/dataset')
import dataset 

def calculator(model, dataset, x_dataset_ohe, y_dataset_ohe, dId_list, year, month, day, hour, computation_range, what_hour, dId, day_type):
    print("current directory : ", os.getcwd())
    cur_dir = os.getcwd()
    if not os.path.isdir(f"{cur_dir}/csv"):
        print('The directory is not present. Creating a new one..')
        os.mkdir(f"{cur_dir}/csv")
    else:
        files = glob.glob(f"{cur_dir}/csv/*.csv")

        for f in files:
            try:
                #f.unlink()
                os.remove(f)
            except OSError as e:
                print("no file exist ")
        # os.remove(f"{cur_dir}/csv/*.csv")  

    print("dataset in test : ", dataset)
    path = ("{dir}/csv".format(dir=cur_dir))
    result_df_final_cc = pd.DataFrame(columns=["DeviceId", "What Hour", "Computation Range", "Predicted Water Consumtion", "r2_score"])
    result_df_final_mae = pd.DataFrame(columns=["DeviceId", "What Hour", "Computation Range", "Predicted Water Consumtion", "Mean Absolout Error"])
    # dataset.drop(["Day_of_Week", "Is_weekend"],axis=1, inplace=True)   
    # print("dataset : ", dataset)
    # ========================
    # result_df = pd.DataFrame(columns=["DeviceId", "What Hour", "Computation Range", "predicted Water Consumtion", "Mean Absolout Error"])
    i = 0
    for dId in dId_list:
        result_df = pd.DataFrame(columns=["DeviceId", "What Hour", "Computation Range", "Predicted Water Consumtion", "Mean Absolout Error", "r2_score"])
        print("Device ID : ", dId)
        
        df_filtered = dataset[dataset['DeviceId'] == dId]
        df_filtered.reset_index(inplace=True, drop=True)
        # print("df_filtered : ", df_filtered)      

        y_df_filtered = df_filtered.loc[:, ["Value"]]
        x_df_filtered = df_filtered.loc[:, ["DeviceId", "Day", "Month", "Year", "hour", "Day_of_Week", "Is_weekend"]]
        
        for duration in computation_range:
            for wh in what_hour:     
                try:
                    # print("x_df_filtered    :   \n", x_df_filtered)           
                    print("what hour : ", wh)
                    print("Computation Range : ", duration)
                    indexHour = x_df_filtered[(x_df_filtered['Year'] == year) &
                    (x_df_filtered["hour"] == hour) & 
                    (x_df_filtered['Month']== month) &                 
                    (x_df_filtered["Day"] == day)].index
                    # print("df_filtered : ", x_df_filtered)
                    # print("df_filtered : ", y_df_filtered)
                    
                    print("indexHour : ", indexHour[0])
                    start_index = indexHour[0] - (duration + wh)




                    # /////////////////////////////////////////////////     

                    x_dataset = x_dataset_ohe[start_index : start_index + duration]

                    y_dataset = y_dataset_ohe[start_index : start_index + duration]
                    # print("x_dataset : ", x_dataset)
                    # print("x_dataset shape : ", x_dataset.shape)
                    # print("x_dataset : ", y_dataset)
                    # print("y_dataset shape : ", y_dataset.shape)
                    # print("x_dataset : ", x_dataset)
                    # print("y_dataset : ", y_dataset)
                    # print("x_dataset : ", x_dataset)
                    # print("x_dataset : ", x_dataset)



                    # /////////////
                    # x_cols = y_df_filtered.columns[:]
                    # x = y_df_filtered[x_cols]

                    # s = GaussRankScaler()
                    # x_ = s.fit_transform(x)
                    # assert x_.shape == x.shape
                    # y_df_filtered[x_cols] = x_
                    # ===============
                    # print('Number of data points in train data:', x)
                    #-----------------------------------Categorical to Binary-----------------------------------------

                    # Train and Test (x,y) / shuffle false because of importance roll of date in our study----------------------
                    # train_x, test_x, train_y, test_y = train_test_split(X_ohe, y, stratify=y, test_size=0.3, shuffle=False)
                    # #################################

                    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, shuffle=False, test_size=0.2, random_state=42)

                    # x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, shuffle=False, test_size=0.2, random_state=42)
                    # print("x_train shape :   \n", x_train.shape)
                    # print("x_train  :   \n", x_train)
                    # x_train = x_train.to_numpy()
                    print("indexHour[0] :   ", indexHour[0])
                    x_predict = x_dataset_ohe[indexHour[0]]
                    # x_predict = x_dataset_ohe.loc[[indexHour[0]]]
                    y_predict = y_dataset_ohe[indexHour[0]]  
                    # print("x_train  :", x_train)
                    # print("y_train  :", y_train)                     
                    # print("x_train shape :", x_train.shape)
                    # print("y_train shape :", y_train.shape) 
                    # print("x_test  :", x_test)
                    # print("y_test  :", y_test)                     
                    # print("x_test shape :", x_test.shape)
                    # print("y_test shape :", y_test.shape) 
                    # print("x_predict :", x_predict)
                    # print("y_predict :", y_predict)
                    # print("x_predict shape :", x_predict.shape)
                    # print("y_predict shape :", y_predict.shape)
                
                                
                    x_predict = np.reshape(x_predict,(1, x_predict.shape[0], 1))
                    print("x_predict shape :", x_predict.shape)
                    x_train=np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))
                    x_test=np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))

                    # def keras_model(input):
                    #     inputs = keras.Input(shape=(input, 1))
                    #     model = layers.LSTM(12, return_sequences=True)(inputs)
                    #     model = layers.LSTM(12)(model)  
                    #     model = layers.Dense(10)(model)
                    #     outputs = layers.Dense(1)(model)
                    #     model = keras.Model(inputs=inputs, outputs=outputs, name="water_predictor")
                    #     return model

                    
                    # model = keras_model(x_train.shape[1])
                    # model.summary()
                    # opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
                    # # model.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae', 'mape', 'cosine'])
                    # model.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae', 'mape'])  

                    # history = model.fit(x_train, y_train,epochs=5, batch_size=50, verbose=1)
                    history = model.fit(x_train, y_train,epochs=2, batch_size=100, verbose=1)
                    # evaluation_dict = neural_network_evaluator.evaluate_ann(history, model, x_train, y_train, x_test, y_test, x_cv, y_cv, x_predict)

                    # x_predict = np.reshape(x_predict,(x_predict.shape[0], 7, 1))  
                    y_pred = model.predict(x_test)
                    print("y_pred : ", y_pred)    
                    pred = model.predict(x_predict)[0] 
                    print("R² score, the coefficient of determination  : ", r2_score(y_test, y_pred))
                    print("history keis :   ", history.history.keys())
                    print("Loss value :   ", history.history['loss'])
                    print("Mean Squared Erro  :   ", history.history['mean_squared_error'])
                    print("Mean Absolute Error  :   ", history.history['mean_absolute_error'])
                    print("Mean Absolute Percentage Error  :   ", history.history['mean_absolute_percentage_error'])



                    # visualiser.plotter(clf, x_train, y_train, x_test, y_test)
                    result_df.at[i, 'DeviceId'] = dId
                    result_df.at[i, "What Hour"] = wh
                    result_df.at[i, "Computation Range"] = duration
                    result_df.at[i, 'Predicted Water Consumtion'] = pred[0]
                    result_df.at[i, 'Mean Absolout Error'] = history.history['mean_absolute_error'][0]
                    result_df.at[i, "r2_score"] = r2_score(y_test, y_pred)
                    i = i + 1
                except Exception as e:
                    # logging.error("something went wrong", exc_info=e)
                    print("there is no value for this device ID : ", dId)
                    
        
        print("path  :  ", path)
        result_df.to_csv(f'{path}\\result_{dId}.csv', index = False)
        # print("result min   :   ", result_df["Mean Absolout Error"].min())   
        print("result_df    :   ", result_df)     
        max_row = result_df[result_df["r2_score"] == result_df["r2_score"].max()]
        print("max row : ", max_row)
        result_df_final_cc = pd.concat([result_df_final_cc,max_row], axis=0)
        # print("final result : ", result_df_final)
        # result_df_final_mae.to_csv(f'{path}\\result_mae_{dId}.csv', index = False)
        # print("result min   :   ", result_df["Mean Absolout Error"].min())        
        min_row = result_df[result_df["Mean Absolout Error"] == result_df["Mean Absolout Error"].min()]
        # print("min row : ", min_row)
        result_df_final_mae = pd.concat([result_df_final_mae,min_row], axis=0)
        # print("final result : ", result_df_final)
       

    result_df_final_cc = result_df_final_cc.reset_index() 
    result_df_final_cc.drop(["index"], axis=1, inplace=True)
    result_df_final_cc = result_df_final_cc.drop_duplicates(subset=['DeviceId'])
    result_df_final_cc.dropna(inplace=True)
    print("result_df_final_cc  after dropna :   \n", result_df_final_cc)
    result_df_final_cc.to_csv(f'{path}\\final_result_cc.csv', index = False)
    result_df_final_mae = result_df_final_mae.reset_index() 
    result_df_final_mae.drop(["index"], axis=1, inplace=True)
    result_df_final_mae = result_df_final_mae.drop_duplicates(subset=['DeviceId'])
    result_df_final_mae.to_csv(f'{path}\\final_result_mae.csv', index = False)

    print(day_type)
    print("final r2_score mean :\n", round(result_df_final_cc.mean(), 4))
    print("final r2_score sum :\n", round(result_df_final_cc.sum(), 4))
    print("final Mean Absolout Error mean :\n", round(result_df_final_mae.mean(), 4))
    print("final Mean Absolout Error sum :\n", round(result_df_final_mae.sum(), 4))
    # print("result : ", result_df_final)
    # pk = result_df_final.drop(["DeviceId", "What Hour", "Predicted Water Consumtion"], axis=1)  
    df = pd.read_csv(f'csv\\result_{dId}.csv')      
    # print("df : ", df)   
    msg = "Chart of miminum MAE of single device ID" 
    df = df.drop(["DeviceId", "Predicted Water Consumtion"], axis=1)
    visualiser.computation_range_plotter_mae(df, msg)
    msg = "Chart of computation of miminum MAE of single device ID" 
    # df = df.drop(["DeviceId", "Predicted Water Consumtion"], axis=1)
    # visualiser.computation_range_mae(df, msg)
    visualiser.computation_range_mae(df, msg)
    msg = "Chart of what hour of miminum MAE of single device ID" 
    # df = df.drop(["DeviceId", "Predicted Water Consumtion"], axis=1)
    visualiser.what_hour_mae(df, msg)

    df = pd.read_csv(f'csv\\final_result_cc.csv')
    # print("df   :   ", df)
    msg = "Chart of maximum of r2_score of all device ID"
    df = df.drop(["DeviceId", "Predicted Water Consumtion"], axis=1)
    visualiser.computation_range_plotter_r2(df, msg)



    df = pd.read_csv(f'csv\\final_result_mae.csv')
    # print("df   :   ", df)
    msg = "Chart of miminum MAE of all device ID"
    df = df.drop(["DeviceId", "Predicted Water Consumtion"], axis=1)
    visualiser.computation_range_plotter_mae(df, msg)

    df = pd.read_csv(f'csv\\final_result_cc.csv')
    # print("df   :   ", df)
    msg = "Chart of maximum of r2_score of all device ID"
    df = df.drop(["DeviceId", "Predicted Water Consumtion"], axis=1)
    visualiser.computation_range_plotter_r2(df, msg)

# ======================= plot for mae 
    pk = result_df_final_mae.drop(["DeviceId", "Computation Range", "Predicted Water Consumtion", "r2_score"], axis=1)
    df = result_df_final_mae.drop(["DeviceId", "Predicted Water Consumtion"], axis=1)   
    
    # gk = pk.groupby(['What Hour'], axis=0).count()    
    gk = pk.groupby(['What Hour'], axis=0).sum()   

 # ========================== test 
    """
    gk.groupby(['What Hour'], axis=0).sum().plot(kind="line", linewidth='2',
                label='MAE',marker="o",
                markerfacecolor="red", markersize=10)
  
    plt.xlabel('What Hour')
    plt.ylabel('Mean Absolout Error')
    plt.title("Chart of sum of miminum MAE of all device ID")
    plt.legend()
    plt.show()
    """
    # gk = gk.groupby(['What Hour'], axis=0).sum()
    visualiser.gb_plotter(pk, 'What Hour', 'Mean Absolout Error', "Chart of sum of miminum MAE of all device ID")

# =======================================================

    pk = result_df_final_mae.drop(["DeviceId", "What Hour", "Predicted Water Consumtion", "r2_score"], axis=1)
    df = result_df_final_mae.drop(["DeviceId", "Predicted Water Consumtion"], axis=1)   
    
    gk = pk.groupby(['Computation Range'], axis=0).count()    
    gk = pk.groupby(['Computation Range'], axis=0).sum()   
    visualiser.gb_plotter(pk, 'Computation Range', 'Mean Absolout Error', "Chart of sum of miminum MAE of all device ID")

# ======================================= plot for r2_score 
    pk = result_df_final_cc.drop(["DeviceId", "Computation Range", "Predicted Water Consumtion", "Mean Absolout Error"], axis=1)
    # pk.dropna(inplace=True)
    # print("pk   :   ", pk)
    df = result_df_final_cc.drop(["DeviceId", "Predicted Water Consumtion"], axis=1)   
    
    gk = pk.groupby(['What Hour'], axis=0).count()    
    gk = pk.groupby(['What Hour'], axis=0).sum()   
    
    try:
        visualiser.gb_plotter(pk, 'What Hour', 'r2_score', "Chart of sum of maximum r2_score of all device ID")

        pk = result_df_final_cc.drop(["DeviceId", "What Hour", "Predicted Water Consumtion", "Mean Absolout Error"], axis=1)
        df = result_df_final_cc.drop(["DeviceId", "Predicted Water Consumtion"], axis=1)   
        
        gk = pk.groupby(['Computation Range'], axis=0).count()    
        gk = pk.groupby(['Computation Range'], axis=0).sum()   
        visualiser.gb_plotter(pk, 'Computation Range', 'r2_score', "Chart of sum of maximum r2_score of all device ID")

    except Exception as e:
        # logging.error("something went wrong", exc_info=e)
        print(e)
        print("There was an error, R2 Score is empty, please select a larger data batch")
       


    
    
        
    


