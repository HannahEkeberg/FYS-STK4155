from logistic_regression import LogisticRegression_class
from creditcard_data import *
from neural_network import *
#from neural_network import *




LogReg = LogisticRegression_class(learning_rate=0.001, batch_size=128, n_epochs=10)

X, y = getData()
X_train, X_test, y_train, y_test = split_data()


#LogReg.sklearn(X,y)

#LogReg.predict_sklearn(X_train, X_test, y_train, y_test)
#LogReg.fit_function(X_train, X_test, y_train, y_test)


LogReg.gradient_descent(X_train,y_train)




"""
NN = NeuralNetwork(X, y, n_hidden_neurons=50, n_categories=10 , epochs=100 , batch_size=100 , eta=0.1 , lmbd=0.0)
#NN.feed_forward()
#print(NN.predict_probabilities(X))
NN.train()

"""
"""
y = y.T[0]
par = np.shape(X)[1] #number of parameters
n = np.shape(X)[0] #number of rows
beta = np.random.randn(par)  #generate random values for beta
"""
#test = LogReg.cost_function(X,y, beta )
#print(test)






if __name__ == "__main__":
    print("Main")
else:
    print("Another program")
