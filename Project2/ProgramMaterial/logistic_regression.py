from sklearn.linear_model import LogisticRegression

import numpy as np
import matplotlib.pyplot as plt
from random import random, seed

from sklearn.metrics import confusion_matrix   #to evaluate the accuracy of a classification.


class LogisticRegression_class:

    def __init__(self, learning_rate, batch_size, n_epochs):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs


    def sigmoid_function(self,x):      #activation function; gives output 0 or 1.
        return(1/(1+np.exp(-x)))

    def gradient_descent(self, X, y,X_val,y_val):   #optimizing parameters beta
        param = np.shape(X)[1] #number of parameters
        n = np.shape(X)[0] #number of rows
        n_batches = n // self.batch_size #integer division
        np.random.seed(42)
        beta = np.random.uniform(0.0,1.0, param)#np.random.randn(param)  #generate random initial values for beta


        error = []
        error_val = []

        index = np.arange(n)
        np.random.shuffle(index)
        batch_split = np.array_split(index,n_batches)


        k = 0
        #total numer of iterations:
        for i in range(self.n_epochs):
            y_pred = self.sigmoid_function(X @ beta)
            y_pred_val = self.sigmoid_function(X_val@beta)
            C = self.cost_function(y, y_pred)
            C_val = self.cost_function(y_val,y_pred_val)
            error.append(C)
            error_val.append(C_val)

            for j in batch_split:
                gradient = np.sum ( X[j].T* (self.sigmoid_function(X[j]@beta)- y[j]),axis=1)

                gradient = gradient/y[j].shape[0]
                beta -= self.learning_rate * gradient
                k += 1

        self.beta = beta
        y_pred = self.sigmoid_function(X @ self.beta)
        tol = 0.5
        y_pred[y_pred >= tol] = 1
        y_pred[y_pred < tol] = 0
        y_pred_val[y_pred_val >= tol] = 1
        y_pred_val[y_pred_val < tol] = 0

        return error, error_val, y_pred, y_pred_val

    def predict_sklearn(self, X_train, X_test, y_train, y_test):
        LR = LogisticRegression(solver='lbfgs', max_iter=400)
        LR.fit(X_train,y_train)
        y_pred = LR.predict(X_test)
        return y_pred

    def cost_function(self, y_data, y_pred):
        cost = -np.mean( (y_data.T * y_pred ) - np.log(1+np.exp(y_pred)))

        return cost
