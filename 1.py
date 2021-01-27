import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import dates as mpl_dates
import sklearn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import seaborn as sns
from sklearn import metrics

pd.set_option('mode.chained_assignment', None)

df = pd.read_csv("Acoustic Logger Data.csv")
###################################################### COLUMN SELECTION LVL & SPR
df_take_Lvl = df.loc[df["LvlSpr"] == "Lvl"]
df_take_Spr = df.loc[df["LvlSpr"] == "Spr"]
###################################################### MELT THE MENTIONED COLUMNS WITH ID & DATE
df_date_Lvl = pd.melt(df_take_Lvl, id_vars=['LvlSpr', 'ID'], value_vars=df.loc[:0, '02-May':].columns.values.tolist(), var_name='Date')
df_date_Spr = pd.melt(df_take_Spr, id_vars=['LvlSpr', 'ID'], value_vars=df.loc[:0, '02-May':].columns.values.tolist(), var_name='Date')
df_merge_LvlSpr = pd.merge(df_date_Lvl, df_date_Spr, on= ['ID', 'Date'], suffixes=("_Lvl", "_Spr"))
df_drop_LvlSpr = df_merge_LvlSpr.drop(['LvlSpr_Lvl', 'LvlSpr_Spr'], axis=1).dropna()
df_drop_LvlSpr['Date'] = pd.to_datetime(df_drop_LvlSpr['Date'], format='%d-%b')
df_drop_LvlSpr['Date'] = df_drop_LvlSpr['Date'].dt.strftime('%d-%m')

df7 = pd.read_csv("Leak Alarm Results.csv")
df7['Date Visited'] = pd.to_datetime(df7['Date Visited'], format='%d/%m/%Y')
df7['Date Visited'] = df7['Date Visited'].dt.strftime('%d-%m')
df_change_column_name = df7.rename(columns={'Date Visited': 'Date'})

df8_merge = pd.merge(df_drop_LvlSpr, df_change_column_name, on=['ID', 'Date'], how='left')
df8_sort = df8_merge.sort_values(['Leak Alarm', 'Leak Found']).reset_index(drop=True)
df8_sort["Leak Alarm"] = df8_sort["Leak Alarm"].fillna("N")

print('OUR FIRST REARRANGED DATASET IS:')
print(df8_sort)                                                         # Dataset before OHE
#################################################### ONE HOT ENCODING
columns_to_OHE = df8_sort[['Date', 'Leak Alarm']]
df11_selected_columns = df8_sort[['ID', 'value_Lvl','value_Spr', 'Leak Found']]
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(columns_to_OHE)
ohe_df = pd.DataFrame(data=onehot_encoded, index=[i for i in range(onehot_encoded.shape[0])],
                      columns=['f'+str(i) for i in range(onehot_encoded.shape[1])])
df8_sort = ohe_df.join(df11_selected_columns)                           # All sorted dataset after OHE
df9 = df8_sort[["Leak Found"]]                                          # All the leak found column of the whole of the dataset
df10 = df8_sort.loc[df9['Leak Found'].notnull()]                        # All 59 labelled given sample with leak found column
# df9["Leak Found"] = df9["Leak Found"].fillna("-1")
# df8.to_csv('OHE.csv')
df10 = df10.drop(labels=['Leak Found'], axis=1)                         # Drop the leak found column of the 59 sample dataset
df12 = df8_sort.drop(labels=['Leak Found'], axis=1)                     # Drop the leak found column of the dataset
################################################### USING SAFE RBF KERNEL FUNCTION FOR GAMMA VARIATION

def rbf_kernel_safe(X, Y=None, gamma=None):
    X, Y = sklearn.metrics.pairwise.check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = sklearn.metrics.pairwise.euclidean_distances(X, Y, squared=True)
    K *= -gamma
    K -= K.max()
    np.exp(K, K)  # exponentiate K in-place
    return K

################################################### LABEL PROPAGATION WITH RBF ALGORITHMS
def label_prop():

    labels = df9.loc[df9['Leak Found'].notnull(), ['Leak Found']]       # label---- All the df9 values labeled as 'Leak found'
    model = LabelPropagation(kernel=rbf_kernel_safe)                    # model called
    model.fit(df10, labels.values.ravel())                              # the 59 available labeled rows are given
    pred = np.array(model.predict(df12))                                # prediction is done on the df12 all unlabelled data
    df13 = pd.DataFrame(pred, columns=['Prediction'])                   # df13----- the previous prediction is shown with prediction column
    df14 = pd.concat([df12, df13], axis=1)                              # df14----- Concatination of the df13 & df12
    print(df14[['ID', 'Prediction']])
    # print(df14.loc[df14['Prediction'] == 'Y'])                         # To show all the rows with leak found 'YES'
    plt.style.use ( 'seaborn' )
    df14['Prediction'].value_counts().plot(kind='bar')
    plt.xticks ( [ 0 , 1 , 2 ] , [ 'NO' , 'YES' , 'N-PRV' ] )
    plt.ylabel('Prediction with RBF algorithm based on number of occurrences (Gamma=None)');
    # plt.show()

# # generate 2 class dataset
# X , y = make_classification ( n_samples=1000 , n_classes=3 , n_features=20 , n_informative=3 , random_state=42 )
#
# # split into train/test sets
# X_train , X_test , y_train , y_test = train_test_split ( X , y , test_size=0.4 , random_state=42 )
#
# # fit model
# clf = OneVsRestClassifier ( LogisticRegression ( ) )
# clf.fit ( X_train , y_train )
# pred = clf.predict ( X_test )
# pred_prob = clf.predict_proba ( X_test )
#
# # roc curve for classes
# fpr = {}
# tpr = {}
# thresh = {}
#
# n_class = 3
#
# for i in range ( n_class ) :
#     fpr [ i ] , tpr [ i ] , thresh [ i ] = roc_curve ( y_test , pred_prob [ : , i ] , pos_label=i )
#
# # plotting
# plt.plot ( fpr [ 0 ] , tpr [ 0 ] , linestyle='--' , color='orange' , label='Class 0 vs Rest' )
# plt.plot ( fpr [ 1 ] , tpr [ 1 ] , linestyle='--' , color='green' , label='Class 1 vs Rest' )
# plt.plot ( fpr [ 2 ] , tpr [ 2 ] , linestyle='--' , color='blue' , label='Class 2 vs Rest' )
# plt.title ( 'Multiclass ROC curve' )
# plt.xlabel ( 'False Positive Rate' )
# plt.ylabel ( 'True Positive rate' )
# plt.legend ( loc='best' )
# plt.savefig ( 'Multiclass ROC' , dpi=300 );
# plt.show();

# # Creating the confusion matrix ##########################################################  NOT WORKING
# cm = metrics.confusion_matrix(df14, pred)
# # Assigning columns names
# cm_df = pd.DataFrame(cm,
#             columns = ['Predicted Negative', 'Predicted Positive'],
#             index = ['Actual Negative', 'Actual Positive'])
# # Showing the confusion matrix
# cm_df
############################################################################################

if __name__ == '__main__':
    label_prop()

################################################## PLOTTING THE REQUIRED COLUMNS

# My_dataset = pd.read_csv('beforeOHE.csv')
# My_dataset.value_Lvl
# print(My_dataset.value_Lvl)

# sns.set(font_scale=1.4)
# My_dataset['value_Lvl'].plot(kind='hist');
# plt.xlabel("Date", labelpad=2)
# plt.ylabel("Average Level (value_Lvl)", labelpad=2)
# plt.title("Distribution of value_Lvl", y=1.015, fontsize=22);
# plt.show()



######################################################## plot
# plt.style.use('seaborn')
#
# data = pd.read_csv('beforeOHE.csv')
# data['Date'] = pd.to_datetime(data['Date'], format='%d-%m')
#
# data.sort_values('Date', inplace=True)
# price_date = data['Date']
# price_close = data['value_Lvl']
# plt.plot_date(price_date, price_close, linestyle='solid')
# plt.gcf().autofmt_xdate()
# date_format = mpl_dates.DateFormatter('%d-%m-%Y')
# plt.gca().xaxis.set_major_formatter(date_format)
# plt.tight_layout()
# plt.title('captured average level (Lvl) histogram on available dates')
# plt.xlabel('Date')
# plt.ylabel('Average level (Lvl)')
# plt.show()

################################################## Histogram for Lvl & Sor
# _ = plt.hist(data['value_Spr'].ravel(), bins='auto')  # arguments are passed to np.histogram
# plt.title("Spread of noise (X-axis) Histogram based on number of occurrences in Y axis ")
#
# plt.show()

############################################ plot after prediction
# data['Leak Found'].value_counts().plot(kind='bar')
# plt.xticks([0,1,2], ['NO', 'YES', 'N-PRV'])
# plt.ylabel('Count');
# plt.show()