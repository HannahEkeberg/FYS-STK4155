import numpy as np

#Cost functions
def neg_log(y_data, y_pred,deriv=False):
    y_pred = y_pred.reshape(y_pred.shape[0],1)
    y_data = y_data.reshape(y_data.shape[0],1)
    if deriv:
        return(y_pred - y_data)
    else:
        return -np.mean( (y_data.T * y_pred) - np.log(1+np.exp(y_pred)))

def MSE(y_data, y_pred, deriv=False):
    if deriv:
        y_pred = y_pred.reshape(y_pred.shape[0],1)
        y_data = y_data.reshape(y_data.shape[0],1)
        return(np.mean(2*(y_pred-y_data)))
    else:
        return(np.mean((y_data-y_pred)**2))




#Activation functions
def tanh(x, deriv=False):
    if deriv:
        tanh = (np.exp(2*x)-1)/(np.exp(2*x)+1)
        return(1-tanh**2)
    else:
        return(np.exp(2*x)-1)/(np.exp(2*x)+1)


def sigmoid_function(x,deriv=False):
    if deriv:
        sig = 1/(1 + np.exp(-x))
        return(sig*(1 - sig))
    else:
        return(1/(1+np.exp(-x)))


def ReLu(x, deriv=False):
    if deriv:
        ReLu = np.maximum(0,x)
        return((x>0)*1)
    else:
        return(np.maximum(0,x))
