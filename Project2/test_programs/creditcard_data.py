import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer


def getData():

    cwd = os.getcwd()  #getting the path of this current program
    filename = cwd + '/default of credit card clients.xls'  #path + file


    ##For all values that are NaN, put into nanDict. Rename column name with space.
    nanDict= {}
    df = pd.read_excel('default of credit card clients.xls', header=1, skiprows=0, index_col=0, na_values=nanDict)
    df.rename(index=str, columns={'default payment next month': 'defaultPaymentNextMonth'}, inplace=True)

    #Drop the rows including data where parameters are out of range
    df = df.drop(df[df.SEX<1].index)
    df = df.drop(df[df.SEX>2].index)
    df = df.drop(df[(df.EDUCATION <1)].index)
    df = df.drop(df[(df.EDUCATION >4)].index)
    df = df.drop(df[df.MARRIAGE<1].index)
    df = df.drop(df[df.MARRIAGE>3].index)

    #Features and targets
    X = df.loc[:, df.columns != 'defaultPaymentNextMonth']
    y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values   #returns array

    onehotencoder = OneHotEncoder(categories='auto')
    #OneHot encoder for column 1,2,3 [sex,education,marriage], increasing the d.o.f.
    #Designmatrix
    X = ColumnTransformer([('onehotencoder', onehotencoder, [1,2,3,5,6,7,8,9,10]),],remainder="passthrough").fit_transform(X)
    #X = ColumnTransformer(
    #[('onehotencoder', onehotencoder, [1,2,3]),],
    #remainder="passthrough").fit_transform(X)
    #y_onehot = onehotencoder.fit_transform(y)
    #print(y_onehot.shape)
    #print(y.shape)
    return X, np.ravel(y)


def split_data():

    X,y = getData()
    seed = 1
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)#, random_state=seed)

    #input scaling of the design matrix
    #sc = StandardScaler()
    #X_train = sc.transform(X_train)
    #X_test = sc.transform(X_test)
    #sc = MinMaxScaler()
    #X_train = sc.fit_transform(X_train)
    #X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test






if __name__ == "__main__":
    print("main:")
else:
    print("creditcard_data:")
