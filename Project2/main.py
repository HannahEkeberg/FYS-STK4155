from logistic_regression import LogisticRegression
from creditcard_data import *


LogReg = LogisticRegression(learning_rate=0.001, n_batches=100, n_epochs=10)
X, y = getData()
print(y)

#LogReg.gradient_descent(X,y)
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
