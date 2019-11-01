from part1 import arrange_data
from sklearn.linear_model import LogisticRegression

import numpy as np
import matplotlib.pyplot as plt
from random import random, seed

from creditcard_data import *




class LogisticRegression:

    def __init__(self, learning_rate, n_batches, n_epochs, lambda_=0, tol=1e-5):
        self.learning_rate = learning_rate
        self.n_batches = n_batches
        self.n_epochs = n_epochs
        self.lambda_ = lambda_
        self.tol = tol



    def sigmoid_function(self,x):
        return(1/(1+np.exp(-x)))

    def gradient_descent(self, X, y):   #optimizing parameters beta
        y = y.T[0]
        par = np.shape(X)[1] #number of parameters
        n = np.shape(X)[0] #number of rows
        beta = np.random.randn(par)  #generate random values for beta
        batch_size  = int(n/self.n_batches)
        #print(batch_size)


        C = 0 #cost function of estimated y. Set to zero before first iterarion.
        error = []


        for i in range(self.n_epochs):   # complete passes through data
            p = self.sigmoid_function(X @ beta)
            #print(p.shape, y.shape, beta.shape)
            #C_model = self.cost_function(p, y, beta)
            #absolute_cost = np.abs(C - C_model)
            #error.append(absolute_cost)
        #print(error)


            #index  = np.arange(n)
            #np.random.shuffle(index)

            #for k in range(self.n_batches):   #number of batches the data is divided into


                #ouch = index[(j+1)*batch_size]
                #print(ouch)
                #random_index = index[j*batch_size:(j+1)*batch_size]
                #Xi = X[random_index, :]
                #yi = y[random_index]

                #gradient  = Xi.T @ (self.sigmoid_function(Xi @ beta)* yi)
        #print(gradient.shape)
                #beta -= self.learning_rate * gradient
                # += 1

        self.X = X
        self.y = y
        self.beta = beta
        #return beta


    def fit_function():
        pass


    def cost_function(self, X,y,beta):

        p = self.sigmoid_function(X @ beta)
        if self.lambda_ == 0:
            Cost = -np.sum(y.T * np.log(p) + (1-y.T)*np.log(1-p) )
        #else:
        #    Cost += np.sum(self.lambda_ * np.sum(beta**2))
        return Cost

    def accuracy_function(self,y_data, y_fit):
        n = np.shape(y_data)[0]
        if np.shape(y_data)[0] == np.shape(y_fit)[0]:
            I = np.where(y_data==y_fit)     ## =1 if y==y, 0 else
        else:
            raise ValuError
        return len(np.ravel(I))/n



if __name__ == "__main__":
    print("main")
else:
    print("logistic_regression:")



"""
Xtrain, Xtest, Y_train_onehot, Y_test_onehot = arrange_data()
#print(np.shape(Y_train_onehot))
class LogisticRegression:

    def __init__(self, learn_rate, iterations, size_minibatch, n ):
        self.learn_rate = learn_rate
        self.n = n  #numb of degrees? batches?
        self.iter = iterations
        self.lambda_ = 0


    def sigmoid_function(self, x):
        return 1./(1+np.exp(-x))

    def cost_function(self, y, X, beta, lambda_):   #max likelihood:

        return -np.sum( y.T * np.log(self.sigmoid(X*beta)) + (1-y.T * np.log(1-self.sigmoid(X*beta)))) + np.sum(self.lmbd*np.sum(self.beta**2))


    def beta(self, p): #number of parameters.
        return np.random.randn(p)

    #def learning_rate(self, t,t0, t1):
    #    return t0/(t+t1)

    def gradient_descent(self,X,y):
        seed( 30 )
        p = np.shape(X)[1] # numb of parameters
        rows = np.shape(X)[0]
        y = np.random.randint(2, size = rows )  #cant understand shape of onehotencoder
        beta = self.beta(p)  #generate initial random beta values
        X_new = X @ beta
        #for i in range(self.iter):
        #    for j in range(size_minibatch):

        gradient = (X_new.T @ (self.sigmoid_function(X_new) - y) - 2*self.lambda_*beta)/self.n

        return gradient





    def fit(self, X, y):
        n_shape = np.shape(X)[1]
        beta = self.get_beta(n_shape)   #parameters to fit!

        for epoch in range(self.n_epochs):
            for i in range(self.iter):
                random_index = np.random.randint(self.iter)




L = LogisticRegression(0.001,2,10, 10)
#var = LogisticRegression(0.1, 4).grad_descent()
#beta = L.get_beta()
grad_descent = L.gradient_descent(Xtrain, Y_train_onehot)
"""
