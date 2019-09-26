import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold



class Regression:

    def __init__(self, method, lamb,X, z):
        self.method = method
        self.lamb = lamb
        self.X = X
        self.z = z

    def OLS(self):
        self.beta_OLS = np.linalg.pinv(self.X.T.dot(self.X)) @ self.X.T.dot(self.z)
        z_model = self.X @ self.beta_OLS
        return z_model


    def Ridge(self):
        n,p=np.shape(self.X)
        I_lambda = np.identity(p, dtype=None)*self.lamb

        self.beta_ridge = np.linalg.inv(self.X.T.dot(self.X) + I_lambda) @ (self.X.T.dot(self.z))
        z_model = self.X @ self.beta_ridge
        return z_model

    def Lasso(self):
        pass



class Error_Analysis:

    def __init__(self, data, model):
        self.data = data
        self.model = model

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
        pass

    def Variance(self):
        pass


class Resampling:

    def __init__(self):
        pass


    def KFold_CrossVal(self, method, poly_degree, x,y,z):
        k = 5
        lamb=0.01
        kf = KFold(n_splits=k, shuffle=True)
        poly = PolynomialFeatures(degree = poly_degree)

        error = np.zeros((k))
        bias = np.zeros((k))
        variance = np.zeros((k))
        R2 = np.zeros((k))
        MSE = np.zeros((k))

        i = 0
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            z_train, z_test = z[train_index], z[test_index]
            X_train = poly.fit_transform(x_train[:,np.newaxis], y_train[:, np.newaxis])

            if method == 'OLS':
                #z_pred = self.Regression().OLS()
                OLS_fit = LinearRegression().fit(X_train, z_train)
                X_test = poly.fit_transform(x_test[:, np.newaxis], y_test[:, np.newaxis])
                z_pred = OLS_fit.predict(X_test)
            elif method == 'Ridge':
                Ridge_fit = Ridge(alpha=lamb).fit(X_train, z_train)
                X_test = poly.fit_transform(x_test[:, np.newaxis], y_test[:, np.newaxis])
                z_pred = Ridge_fit.predict(X_test)
            elif method == 'Lasso':
                Lasso_fit = Lasso(alpha=lamb).fit(X_train, z_train)
                X_test = poly.fit_transform(x_test[:, np.newaxis], y_test[:, np.newaxis])
                z_pred = Lasso_fit.predict(X_test)


            error[i] = np.mean((z_test - z_pred)**2)
            bias[i] = np.mean((z_test - np.mean(z_pred))**2)
            variance[i]  = np.var(z_pred) #RIKTIG?
            R2[i] = 1 - ( np.sum((z_test - z_pred)**2) / np.sum((z_test-np.mean(z_pred))**2) )   ##Riktig???
            MSE[i] = np.sum((z_pred - z_test)**2)/np.size(z_pred) ###MSE??

            i += 1

        #print("Error:", np.mean(error))
        #print("Bias:", np.mean(bias))
        #print("Variance:", np.mean(variance))
        #print("R2:", np.mean(R2))
        #print("MSE:", np.mean(MSE))
        return np.mean(error), np.mean(bias), np.mean(variance), np.mean(R2), np.mean(MSE)







#print(Functions('Ridge', lamb=0.001).Ridge())

        #beta = (np.linalg.pinv(X.T.dot(X)) @  X.T.dot(z))
        #z_model = X @ beta
