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

    def __init__(self, model, lamb, X, z):
        self.model = model
        self.lamb = lamb
        self.X = X
        self.z = z

    def OLS(self):
        self.beta_OLS = np.linalg.pinv(self.X.T.dot(self.X)) @ self.X.T.dot(self.z)
        #z_model = self.X @ self.beta_OLS
        #return z_model


    def Ridge(self):
        n,p=np.shape(self.X)
        I_lambda = np.identity(p, dtype=None)*self.lamb

        self.beta_ridge = np.linalg.inv(self.X.T.dot(self.X) + I_lambda) @ (self.X.T.dot(self.z))
        #z_model = self.X @ self.beta_ridge
        #return z_model

    def Lasso(self):
        self.beta_Lasso = Lasso(alpha=self.lamb).fit(self.X, self.z).coef_
        #X_test = poly.fit_transform(x_test[:, np.newaxis], y_test[:, np.newaxis])
        #z_model = Lasso_fit.predict(self.X)
        #return z_model


    def predict(self):
        if self.model == 'OLS':
            self.OLS()  #
            return self.X @ self.beta_OLS
        elif self.model == 'ridge':
            self.Ridge()
            return X @ self.beta_ridge
        elif self.model == 'Lasso':
            self.Lasso()
            return X @ self.beta_Lasso

class Error_Analysis:

    def __init__(self, data, model):
        self.data = data
        self.model = model

    def Error(self):
        return np.mean((self.data - self.model)**2)

    def MSE(self):
        n = len(self.data)
        #data = self.data
        #model = self.model
        MSE = 1/n * np.sum((self.data-self.model)**2)
        return MSE

    def R2(self):
        R2 = 1 - ( np.sum((self.data-self.model)**2) / np.sum((self.data-np.mean(self.model))**2) )

    def RelativeError(self):
        return np.abs((self.data-self.model)/self.data)

    def Bias(self):
        return np.mean((self.data - self.model)**2)

    def Variance(self):
        return np.var(self.model)


class Resampling:

    def __init__(self):
        pass
        #self.model = model


    def KFold_CrossVal(self, model, poly_degree,X, x,y,z):
        #model = self.model
        k = 5
        lamb=0.01
        kf = KFold(n_splits=k, shuffle=True)
        poly = PolynomialFeatures(degree = poly_degree)

        error_train = np.zeros((k)); error_test = np.zeros((k))
        bias_train = np.zeros((k)); bias_test = np.zeros((k))
        variance = np.zeros((k))
        R2 = np.zeros((k))
        MSE = np.zeros((k))
        #mtd = Regression('OLS', 0.1, X, z)

        i = 0
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            z_train, z_test = z[train_index], z[test_index]
            X_train, X_test = X[train_index], X[test_index]
            #X_train = poly.fit_transform(x_train[:,np.newaxis], y_train[:, np.newaxis])

            if model == 'OLS':
                zPred_Tr = Regression('OLS', 0, X_train, z_train).predict()
                zPred_Te = Regression('OLS', 0, X_test, z_test).predict()
                #OLS_fit = LinearRegression().fit(X_train, z_train)
                #X_test = poly.fit_transform(x_test[:, np.newaxis], y_test[:, np.newaxis])
                #z_pred_Tr = OLS_fit.predict(X_train)
                #z_pred_Te = OLS_fit.predict(X_test)

            elif model == 'Ridge':
                zPred_Tr = Regression('Ridge', lamb, X_train, z_train).predict()
                zPred_Te = Regression('Ridge', lamb, X_test, z_test).predict()
                print(zPred_Tr)
                #Ridge_fit = Ridge(alpha=lamb).fit(X_train, z_train)
                #X_test = poly.fit_transform(x_test[:, np.newaxis], y_test[:, np.newaxis])
                #z_pred = Ridge_fit.predict(X_test)
            elif model == 'Lasso':
                zPred_Tr = Regression('Lasso', lamb, X_train, z_train).predict()
                zPred_Te = Regression('Lasso', lamb, X_test, z_test).predict()

                #Lasso_fit = Lasso(alpha=lamb).fit(X_train, z_train)
                #X_test = poly.fit_transform(x_test[:, np.newaxis], y_test[:, np.newaxis])
                #z_pred = Lasso_fit.predict(X_test)
            else:
                print("Choose valid method")

            error_train[i] = Error_Analysis(z_train, zPred_Tr).Error()
            error_test[i] =  Error_Analysis(z_test, zPred_Te).Error()
            bias_train[i] = Error_Analysis(z_train, zPred_Tr).Bias()
            bias_test[i] = Error_Analysis(z_test, zPred_Te).Bias()


            #bias_train[i] = np.mean(np.mean((z_train - np.mean(zPred_Tr))**2), axis=1, keepdims=True)
            #variance[i]  = np.var(z_pred) #RIKTIG?
            #R2[i] = 1 - ( np.sum((z_test - z_pred)**2) / np.sum((z_test-np.mean(z_pred))**2) )   ##Riktig???
            #MSE[i] = np.sum((z_pred - z_test)**2)/np.size(z_pred) ###MSE??

            i += 1

        #print("Error:", np.mean(error))
        #print("Bias:", np.mean(bias))
        #print("Variance:", np.mean(variance))
        #print("R2:", np.mean(R2))
        #print("MSE:", np.mean(MSE))
        #return np.mean(error), np.mean(bias), np.mean(variance), np.mean(R2), np.mean(MSE)
        return error_train, error_test, bias_train, bias_test

    def Bootstrap(self, x,y, z,X ):
        n=40; n_bootstraps=100; maxdegree=14

        error = np.zeros(maxdegree)
        bias = np.zeros(maxdegree)
        variance = np.zeros(maxdegree)
        polydegree = np.zeros(maxdegree)

        xTrain, xTest, yTrain, yTest, zTrain, zTest, XTrain, XTest = train_test_split(x,y,z,X, test_size=0.2)
        print(len(xTrain), len(xTest))
        for degree in range(maxdegree):
            model = make_pipeline(PolynomialFeatures(degree=degree))
            zPred = np.empty((zTest.shape[0], n_bootstraps))
            for i in range(n_bootstraps):
                X_,z_ = resample(XTrain,zTrain)  #z_: shape=16,20, Z_: shape=16,6

                zPred[:,i] = 1
            #zPred[:, i] = model.fit(X_, z_)#.predict(XTest).ravel()   #zPred[:,i] = model.fit(X_,z_).predict(XTest).ravel()

        #polydegree[degree]=degree
        #error[degree] = np.mean( np.mean((zTest - zPred)**2, axis=1, keepdims=True) )
        #bias[degree] = np.mean( (zTest - np.mean(zPred, axis=1, keepdims=True))**2 )
        #variance[degree] = np.mean( np.var(zPred, axis=1, keepdims=True) )

        #print('Polynomial degree:', degree)
        #print('Error:', error[degree])
        #print('Bias^2:', bias[degree])
        #print('Var:', variance[degree])
        #print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))
        #return polydegree, error, bias, variance
        return zPred
