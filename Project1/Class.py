import numpy as np
from numpy.random import randint, randn

import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab

import pandas as pd

from time import time

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample


"""
Program containts 3 different classes, regression for OLS, ridge and lasso, error analysis and resampling for K-fold cross validation and 
bootstrap. 

The regression class takes in arguments; model, designmatrix X and data z when calling on the class. In the function predict,
the predicted data is predicted with a design matrix of choice. The program is a little tricky, because it is hard to vary values of lambda for
ridge and lasso.

The error class simply takes model and data as argument and predicts the error thereby. 

The resampling class takes model, design matrix X, variable x and variable y, z (f(x,y)) as arguments. Thereby, one can choose method
K fold cross validation or bootstrap, which returns the error analysis on the train and test data. 

"""



class Regression:

    def __init__(self, model, X,z):
        self.model = model
        #self.lamb = lamb
        self.X = X
        self.z = z


    def OLS(self, get_beta=False):
        self.beta_OLS = np.linalg.pinv(self.X.T.dot(self.X)) @ self.X.T.dot(self.z)
        if get_beta==True:
            return self.beta_OLS



    def Ridge(self, lamb=1e-9, get_beta=False):
        n,p=np.shape(self.X)
        I_lambda = np.identity(p, dtype=None)*lamb
        self.beta_ridge = np.linalg.inv(self.X.T.dot(self.X) + I_lambda) @ (self.X.T.dot(self.z))
        if get_beta==True:
            return self.beta_ridge


    def Lasso(self, lamb=1e-5, get_beta=False):
        self.beta_Lasso = Lasso(alpha=lamb).fit(self.X, self.z).coef_
        if get_beta==True:
            return self.beta_Lasso

    def predict(self, X):
        if self.model == 'OLS':
            self.OLS()
            return X @ self.beta_OLS
        elif self.model == 'Ridge':
            self.Ridge()
            return X @ self.beta_ridge
        elif self.model == 'Lasso':
            self.Lasso()
            return X @ self.beta_Lasso



class Error:

    def __init__(self, data, model):
        self.data = data
        self.model = model

    def MSE(self):
        n = len(self.data)
        return 1/n * np.sum((self.data-self.model)**2)

    def R2(self):
        return 1 - ( np.sum((self.data-self.model)**2) / np.sum((self.data-np.mean(self.model))**2) )

    def bias(self):
        return np.mean((self.data - np.mean(self.model))**2)

    def Var(self):
        return np.var(self.model)



class Resample:

    def __init__(self, model, X, x, y, z):
        self.model = model
        self.X = X#.astype('float64')
        self.x = x
        self.y = y
        self.z = z#.astype('float64')

    def KFold_CrossVal(self,k):

        kf = KFold(n_splits=k, shuffle=True)
        #poly = PolynomialFeatures(degree = poly_degree)

        #X_new = poly.fit_transform(self.X) #changing design matrix to appropriate numb of degrees

        error = np.zeros((k)); error_test = np.zeros((k))
        bias = np.zeros((k)); bias_test = np.zeros((k))
        variance = np.zeros((k))
        R2_train = np.zeros((k))
        R2_test = np.zeros((k))
        MSE_test = np.zeros((k))
        MSE_train = np.zeros((k))

        i = 0
        for train_index, test_index in kf.split(self.z): #splts len(z) times?
            x_train, x_test = self.x[train_index], self.x[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            z_train, z_test = self.z[train_index], self.z[test_index]
            X_train, X_test = self.X[train_index], self.X[test_index]

            if self.model == 'OLS':
                zPred_Te = Regression('OLS', X_train, z_train).predict(X_test)
                zPred_Tr = Regression('OLS', X_train, z_train).predict(X_train)
            elif self.model == 'Ridge':
                zPred_Te = Regression('Ridge', X_train, z_train).predict(X_test)
                zPred_Tr = Regression('Ridge', X_train, z_train).predict(X_train)
            elif self.model == 'Lasso':
                zPred_Te = Regression('Lasso', X_train, z_train).predict(X_test)
                zPred_Tr = Regression('Lasso', X_train, z_train).predict(X_train)
            else:
                return "Choose valid method"


            R2_test[i] = Error(z_test, zPred_Te).R2()
            R2_train[i] = Error(z_train, zPred_Tr).R2()
            MSE_test[i] = Error(z_test, zPred_Te).MSE()
            MSE_train[i] = Error(z_train, zPred_Tr).MSE()
            i +=1

        else:
            return np.mean(R2_test), np.mean(R2_train), np.mean(MSE_test), np.mean(MSE_train)
  

    def Bootstrap(self, n_bootstraps):
        z_train, z_test, X_train, X_test = train_test_split(self.z, self.X, test_size=0.2)
        sampleSize = X_train.shape[0]
        z_pred = np.empty((z_test.shape[0], n_bootstraps))
        z_train_pred = np.empty((z_train.shape[0], n_bootstraps))
        z_train_boot = np.empty((z_train.shape[0], n_bootstraps))

        for i in range(n_bootstraps):
            X_,z_ = resample(X_train, z_train)
            if self.model == 'OLS':
                model = Regression('OLS', X_, z_)
            elif self.model == 'Ridge':
                model = Regression('Ridge', X_, z_)
            elif self.model == 'Lasso':
                model = Regression('Lasso', X_, z_)
            else:
                print("Choose valid method")


            z_pred[:,i] = model.predict(X_test)#.ravel()
            z_train_pred[:,i] = model.predict(X_train)
            z_train_boot[:,i] = z_

        MSE_test = np.mean( np.mean((z_test[:, None] - z_pred)**2, axis=1, keepdims=True) )
        #MSE_train = np.mean( np.mean((z_train_pred[:, None] - z_train_boot)**2, axis=1, keepdims=True))
        MSE_train = np.mean( np.mean((z_train[:, None] - z_train_pred)**2, axis=1, keepdims=True))
        bias = np.mean( (z_test[:,None] - np.mean(z_pred, axis=1, keepdims=True))**2 )
        Var = np.mean( np.var(z_pred, axis=1, keepdims=True) )

        return MSE_test, MSE_train, bias, Var

