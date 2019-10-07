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



class Regression:

    def __init__(self, model, X,z):
        self.model = model
        #self.lamb = lamb
        self.X = X
        self.z = z

    #def X(self, x,y, degree=6):
    #    X_ = np.c_[x,y]
    #    poly = PolynomialFeatures(degree=degree)
    #    self.X = poly.fit_transform(X_)

    #def OLS(self):
    #    self.beta_OLS = np.linalg.pinv(self.X.T.dot(self.X)) @ self.X.T.dot(self.z)



    def OLS(self, get_beta=False):
        self.beta_OLS = np.linalg.pinv(self.X.T.dot(self.X)) @ self.X.T.dot(self.z)
        #print("betashape", np.shape(self.beta_OLS))
        #print(self.beta_OLS)
        if get_beta==True:
            return self.beta_OLS



    def Ridge(self, lamb=0.01, get_beta=False):
        n,p=np.shape(self.X)
        I_lambda = np.identity(p, dtype=None)*lamb
        self.beta_ridge = np.linalg.inv(self.X.T.dot(self.X) + I_lambda) @ (self.X.T.dot(self.z))
        if get_beta==True:
            return self.beta_ridge


    def Lasso(self, lamb=0.01, get_beta=False):
        self.beta_Lasso = Lasso(alpha=lamb).fit(self.X, self.z).coef_
        if get_beta==True:
            return self.beta_Lasso



    def predict(self, X):
        if self.model == 'OLS':
            self.OLS()
            return X @ self.beta_OLS
        elif self.model == 'ridge':
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

    def __init__(self, model, X, x, y,z):
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
        R2 = np.zeros((k))
        MSE_test = np.zeros((k))
        MSE_train = np.zeros((k))

        i = 0
        for train_index, test_index in kf.split(self.z): #splts len(z) times?
            print(kf.split(self.z))
            x_train, x_test = self.x[train_index], self.x[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            z_train, z_test = self.z[train_index], self.z[test_index]
            X_train, X_test = self.X[train_index], self.X[test_index]

            if self.model == 'OLS':
                zPred_Te = Regression('OLS', X_train, z_train).predict(X_test)
                zPred_Tr = Regression('OLS', X_train, z_train).predict(X_train)
                #return zPred_Te
            elif self.model == 'Ridge':
                zPred_Te = Regression('Ridge', X_train, z_train).predict(X_test)
            elif self.model == 'Lasso':
                zPred_Te = Regression('Lasso', X_train, z_train).predict(X_test)
            else:
                return "Choose valid method"

            error[i] = np.mean((z_test-zPred_Te)**2)
            bias[i] = Error(z_test, zPred_Te).bias()
            variance[i] = Error(z_test, zPred_Te).Var()
            R2_test[i] = Error(z_test, zPred_Te).R2()
            R2_train[i] = Error(z_train, zPred_Tr).R2()
            MSE_test[i] = Error(z_test, zPred_Te).MSE()
            MSE_train[i] = Error(z_train, zPred_Tr).MSE()
            i +=1


        return np.mean(error), np.mean(bias), np.mean(variance), np.mean(R2_test), np.mean(R2_train), np.mean(MSE_test), np.mean(MSE_train)

    def Bootstrap(self, n_bootstraps):
        #error= np.zeros((k)); error_test = np.zeros((k))
        #bias = np.zeros((k)); bias_test = np.zeros((k))
        #variance = np.zeros((k))
        #R2 = np.zeros((k))
        #MSE = np.zeros((k))

        z_train, z_test, X_train, X_test = train_test_split(self.z, self.X, test_size=0.2)
        #print("ztrain shape:", np.shape(z_train))
        #print("ztest shape:", np.shape(z_test))
        sampleSize = X_train.shape[0]
        z_pred = np.empty((z_test.shape[0], n_bootstraps))
        #print("z pred shape: ", np.shape(z_pred))
        z_train_pred = np.empty((z_train.shape[0], n_bootstraps))
        z_train_boot = np.empty((z_train.shape[0], n_bootstraps))
        #z = self.z.reshape(-1,1)

        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
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

            #z_train_boot[:,i] = model.predict(X_test).ravel()  #model.predict(X_test)
            #print("Shape z pred in bootstrap" ,np.shape(z_pred), i)
            #z_train_boot = z_.ravel()        #model.predict(X_test)

            z_pred = model.predict(X_test)#.ravel()
            #print(z_pred)
            #print(np.shape(z_pred))
        #error = Error(z_test, z_pred).error()
        MSE = Error(z_test, z_pred).MSE()
        R2 = Error(z_test, z_pred).R2()
        bias = Error(z_test, z_pred).bias()
        Var = Error(z_test, z_pred).Var()
        return MSE, R2, bias, Var#np.mean(MSE), np.mean(R2), np.mean(bias), np.mean(Var)
            #z_pred_train = model.predict(X_train).ravel()

            #indices = np.random.randint(0, sampleSize, sampleSize)
            #X_, z_ = X_train[indices], z_train[indices]
            #z_train_boot[:,i] = z_

            #x_, y_, z_ = resample(x_train, y_train, z_train)
            #X_new = np.c_[x_test, y_test]
            #poly = PolynomialFeatures(degree=6)
            #X_test = poly.fit_transform(X_new)

            #if self.model == 'OLS':
            #model_ = Regression('OLS', X_, z_).OLS().predict(X_test)
                #z_pred = model_.predict(X_test)
        #print(z_pred)
        #z_test = z_test.reshape((len(z_test), 1))
        #print(np.shape(z_test))
        #print(z_test.ravel())
        #error = np.mean( np.mean((z_pred - z_test)**2, axis=1, keepdims=True))
        #print(np.shape(error))
        #error = np.mean( np.mean((z_pred - z_test)**2, axis=1, keepdims=True))


        #return error




        #for degree in range(max_degree):
            #poly = PolynomialFeatures(degree=degree)
            #X_new = poly.fit_transform(self.X)

            #OLS = Regression('OLS', self.X, self.z)
            #m = make_pipeline(PolynomialFeatures(degree=degree), OLS)


            #model = make_pipeline(PolynomialFeatures(degree=degree), OLS)
            #y_pred = np.empty((y_test.shape[0], n_bootstraps))
            #for i in range(n_bootstraps):
        #        x_, y_ = resample(x_train, y_train)
        #        y_pred[:,i] = model.fit(x_, y_).predict(x_test).ravel()

        #    polydegree[degree]=degree
        #    error[degree] = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
        #    bias[degree] = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
        #    variance[degree] = np.mean( np.var(y_pred, axis=1, keepdims=True) )

        #return polydegree, error, bias, variance
        #return 0
        #return OLS



