
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from random import random, seed

from sklearn.metrics import confusion_matrix   #to evaluate the accuracy of a classification.



#from creditcard_data import *




class LogisticRegression_class:

    def __init__(self, batch_size, n_epochs, lambda_=0, tol=1e-5):
        #self.learning_rate = learning_rate
        #self.n_batches = n_batches
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lambda_ = lambda_
        self.tol = tol



    def sigmoid_function(self,x):
        return(1/(1+np.exp(-x)))

    def gradient_descent(self, X, y):   #optimizing parameters beta
        y = y.T
        par = np.shape(X)[1] #number of parameters
        n = np.shape(X)[0] #number of rows
        n_batches = n // self.batch_size #integer division
        beta = np.random.randn(par)  #generate random initial values for beta




        error = []

        n_0_list = []
        n_1 = np.count_nonzero(y)   #total number of y=1. double check that those are really 1.
        #const = (n - n_1)/n_1

        error = []
        k = 0
        #total numer of iterations:
        for i in range(self.n_epochs):
            y_pred = self.sigmoid_function(X @ beta)
            #print(y_pred)
            C = self.cost_function(y, y_pred)
            #abs_C  = np.abs(C_data - C)
            error.append(C)

            for j in range(n_batches):
                #gange gradienten med en verdi C=antall default ==0 and default==1.
                gradient  = np.sum(X.T * self.sigmoid_function(X @ beta) - y*X) #X.T @ (self.sigmoid_function(X @ beta) * const * y)  #const*0 =0, const*1 =const, gives higher val for undrerepresented 1's
                gradient = gradient/self.batch_size
                #print(gradient.shape)
                learning_rate = self.learning_schedule(i*n_batches+j)     #adaptive learning rate. Sjekke om det enkelste funker forst.
                beta -= learning_rate * gradient
                k += 1

        self.beta = beta
        y_pred = self.sigmoid_function(X @ self.beta)
        tol = 0.5
        y_pred[y_pred >= tol] = 1
        y_pred[y_pred < tol] = 0

        I = accuracy_function(y, y_pred)

        return error, self.n_epochs

    def learning_schedule(self,t):  #??? what is this
        t0 = 5; t1 = 50
        return t0/(t+t1)

    def fit_function(self, X_train, X_test, y_train, y_test):
        self.gradient_descent(X_train, y_train)
        y_pred = X_test @ self.beta


    def predict_sklearn(self, X_train, X_test, y_train, y_test):
        LR = LogisticRegression()
        LR.fit(X_train,y_train)
        y_pred = LR.predict(X_test)
        v = np.count_nonzero(y_pred)
        I = self.accuracy_function(y_test, y_pred)
        C = self.cost_function(y_test, y_pred)
        print(I)
        CM = self.accuracy_matrix(y_test, y_pred, 'sklearn')

    def cost_function(self, y_data, y_pred):  #oppdatere parametrene.
        return np.sum( (y_data.T * y_pred ) - np.log(1+np.exp(y_pred)))
        #return -np.sum(y_data.T)* np.log(y_pred) + (1-y_data.T)*np.log(1-y_pred))

    def accuracy_function(self,y_data, y_pred):  #hvor bra modellen gjor det.
        if np.shape(y_data)[0] == np.shape(y_pred)[0]:
            I = np.mean(y_data==y_pred)     ## =1 if y==y, 0 else.... np.mean #dobbelsjekk at det funker
        else:
            raise ValuError
        return I

    def accuracy_matrix(self, y_data, y_pred, title):
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_data, y_pred)
        print(cm)
        cmap = plt.cm.Blues
        im = ax.imshow(cm, interpolation = 'nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks = np.arange(cm.shape[1]),
               yticks = np.arange(cm.shape[0]),
               title=title,
               xlabel='predicted y value',
               ylabel='original y value')
        plt.show()

        return cm






if __name__ == "__main__":
    print("main")
else:
    print("logistic_regression:")
