"""
Idunn Mostue - Hannah Ekeberg



"""


import pandas as pd
import os
import numpy as np
import matplotlib.pylab as plt
#import matplotlib.pyplot as plt
#%matplotlib inline

#import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

def getData():
    cwd = os.getcwd()  #getting the path of this current program
    filename = cwd + '/default of credit card clients.xls'  #path + file

    np.random.seed(0)
    #Read file into pandas dataframe
    nanDict= {}
    df = pd.read_excel('default of credit card clients.xls', header=1, skiprows=0, index_col=0, na_values=nanDict)
    df.rename(index=str, columns={'default payment next month': 'defaultPaymentNextMonth'}, inplace=True)

    #Drop the rows including data where parameters are out of range
    df=df.drop(df[df.SEX<1].index)
    df=df.drop(df[df.SEX<2].index)
    df=df.drop(df[(df.EDUCATION <1)].index)
    df=df.drop(df[(df.EDUCATION >4)].index)
    df=df.drop(df[df.MARRIAGE<1].index)
    df=df.drop(df[df.MARRIAGE>3].index)

    #Drop the rows for the customers that do not have any bills & corresponding payments throughout the period
    #as we do not want the model to train on this data
    df = df.drop(df[(df.BILL_AMT1 == 0) &
                (df.BILL_AMT2 == 0) &
                (df.BILL_AMT3 == 0) &
                (df.BILL_AMT4 == 0) &
                (df.BILL_AMT5 == 0) &
                (df.BILL_AMT6 == 0) &
                (df.PAY_AMT1 == 0) &
                (df.PAY_AMT2 == 0) &
                (df.PAY_AMT3 == 0) &
                (df.PAY_AMT4 == 0) &
                (df.PAY_AMT5 == 0) &
                (df.PAY_AMT6 == 0)].index)

    #Features (X) and targets (y):
    #Divide features into categorical and continous data
    X_categorical = df.iloc[:,[1,2,3,5,6,7,8,9,10]].values.copy()
    X_continous = df.iloc[:,[0,4,11,12,13,14,15,16,17,18,19,20,21,22]].values.copy()

    y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values.copy()


    #OneHot encoder for categorial data -> [sex,education,marriage, pay_april, pay_may, pay_jun, pay_jul, pay_aug, pay_sep]
    onehotencoder = OneHotEncoder(categories="auto")
    X_categorical = ColumnTransformer([('onehotencoder', onehotencoder, [0,1,2,3,4,5,6,7,8]),],remainder="passthrough").fit_transform(X_categorical).todense()
    y = np.ravel(y)

    return X_categorical, X_continous, np.ravel(y), df

def counter_plot():
    #plot
    import seaborn as sns
    X_cat,X_con ,y, df = getData()
    sns.set(style="white")
    sns.set(style="whitegrid")
    sns.countplot(x='defaultPaymentNextMonth',data=df, palette='Set2')
    plt.title('Count of default payment next month (NO=0, YES=1)')
    plt.savefig('counter_plot_balance.png', dpi=300)
    plt.show()

    #percentage calculation
    count_no_default = len(df[df['defaultPaymentNextMonth']==0])
    count_default = len(df[df['defaultPaymentNextMonth']==1])
    pct_of_no_default = count_no_default/(count_no_default+count_default)
    print("percentage of no default payment (NO=0) is", pct_of_no_default*100)
    pct_of_default = count_default/(count_no_default+count_default)
    print("percentage of default payment (YES=1) is", pct_of_default*100)


def balanced_data():
    X_categorical,X_continous,y,df = getData()

    #Fix the imbalance in the data
    ones_  = np.argwhere(y==1).flatten()
    zeros_ = np.argwhere(y==0).flatten()

    indexes = np.random.choice(zeros_,size=ones_.shape[0]) #randomly choose a no. of target points == 0, corresponding to no. of target points ==1
    indexes = np.concatenate((indexes,ones_),axis=0)

    y             = y[indexes]
    X_categorical = X_categorical[indexes]
    X_continous   = X_continous[indexes]
    indexes       = np.arange(y.shape[0])

    np.random.shuffle(indexes)
    y             = y[indexes]
    X_categorical = X_categorical[indexes]
    X_continous   = X_continous[indexes]
    X_continous   = np.array(X_continous)
    X_categorical = np.array(X_categorical)


    #Train-Test Split - Dividing into Train and Validation(test) data 80:20
    trainingShare = 0.8
    seed = 1
    X_train_cat, X_test_cat, X_train_con, X_test_con, y_train, y_test=train_test_split(X_categorical, X_continous, y,
                                                  test_size = 1-trainingShare,
                                                  random_state=seed,shuffle=True)



    #Input Scaling
    sc          = StandardScaler()
    X_train_con = sc.fit_transform(X_train_con)
    X_test_con  = sc.transform(X_test_con)


    #Reunite the categorical and contonous data in the test and validation(test) data
    X_train = np.zeros((X_train_con.shape[0],X_train_con.shape[1] + X_train_cat.shape[1]))
    X_test  = np.zeros((X_test_con.shape[0],X_test_con.shape[1] + X_test_cat.shape[1]))


    X_train[: , X_train_con.shape[1]:] = X_train_cat
    X_train[:,:X_train_con.shape[1]]   = X_train_con

    X_test[:,:X_test_con.shape[1]] = X_test_con
    X_test[:,X_test_con.shape[1]:] = X_test_cat



    return X_train, X_test, y_train, y_test
