import numpy as np
import matplotlib.pyplot as plt

#Produce layers for the neural network, no. of layers are chosen in NeuralNetwork class
# class Dense initialize weights and biases for every layer, and returns the weighted sum.
class Dense:
    def __init__(self,n_inputs,n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.w = np.random.randn(self.n_inputs,self.n_outputs)*np.sqrt(1/(n_inputs+n_outputs)) #initialize for at verdiene ikke skal skli ut til ytterpunktene men holde seg paa midten
        self.b = np.zeros((1,self.n_outputs)) + 0.01

    def __call__(self,x):
        self.output = x@self.w + self.b
        return(self.output)

#
class NeuralNetwork:
    def __init__(self,layers, n_inputs, n_outputs, activation_function, cost_function):
        self.n_inputs         = n_inputs            #Initial X parameters from the data set
        self.n_outputs        = n_outputs           #Outputs in final layer

        self.act_func         = activation_function #activation function
        self.cost_func        = cost_function       #cost function

        self.layers = []
        for i in range(len(layers)):   #layers: hidden layers, with nodes in eg. [10,20]
            self.layers.append(Dense(n_inputs,layers[i]))   #layers[i]=output for current layer
            n_inputs = layers[i]   #n_inputs update --> last layer output is new input
        self.layers.append(Dense(layers[-1],n_outputs))  #last layer appended



    def feed_forward(self,X):
        x = X.copy()
        for i,layer in enumerate(self.layers):          #feed forward by updating current X, and go through activation function
            x = self.act_func[i](self.layers[i](x))    #X--> a^l-> a^(l+1)-->..--> a^L
        return(x)


    def backpropagation(self,X,y):
        x = X.copy()
        y_pred = self.feed_forward(x)      #y_pred --> aL = self.feed-forward(x)

        delta = self.cost_func(y,y_pred, deriv=True)*self.act_func[-1](self.layers[-1].output, deriv=True) #delta^L --> error in output layer




        w_grad = []
        b_grad = []
        b_grad.append(np.sum(delta,axis=0))   #initial change in cost as function of bias

        w_grad.append((self.act_func[-2](self.layers[-2].output.T@delta)))#initial change in cost as function of weight


        l = len(self.layers) - 2                                  #l = L last hidden layer, from which we propagate backwards

        for layer in reversed(self.layers[:-1]):
            #propagating backwards

            delta = (delta@self.layers[l+1].w.T)*self.act_func[l](self.layers[l].output,deriv=True) #delta^l -->(l=L-i) error in hidden layer i

            b_grad.append(np.sum(delta,axis=0))#fetching beta_gradient

            if l==0:

                a = x.T
                grad = a @ delta
            else:
                a = self.act_func[l-1](self.layers[l-1].output)
                grad = a.T @ delta

            w_grad.append(grad)#fetchings weight_gradient
            l-=1


        w_grad = list(reversed(w_grad))                         #Revers list so it starts at input
        b_grad = list(reversed(b_grad))

        return(w_grad,b_grad)

    def train(self,X,y,X_test,y_test,lr,batch_size,epochs):
        error = []
        error_test = []
        data_ind = np.arange(X.shape[0])                           #no. of data points
        iterations = int(X.shape[0] / batch_size)                  #no. of iterations

        #indx = np.arange(X.shape[0])  ###
        #np.random.shuffle(indx)       ###
        #batch_split = np.array_split(indx,iterations) ###

        for i in range(epochs):  #gradient descent

            for j in range(iterations):
            #for j in batch_split:  ###
                #pick datapoints with replacements
                chosen_datapoints = np.random.choice(data_ind, size=batch_size, replace=False)


                #minibatch data
                X_full = X[chosen_datapoints]
                y_full = y[chosen_datapoints]

                w_grad,b_grad = self.backpropagation(X_full,y_full)

                for index, layer in enumerate(self.layers): #update gradient descent

                    self.layers[index].w = self.layers[index].w - lr*w_grad[index]/batch_size
                    self.layers[index].b = self.layers[index].b - lr*b_grad[index]/batch_size


            y_pred = self.feed_forward(X)
            cost_ = self.cost_func(y,y_pred,deriv=False)
            error.append(cost_)
            y_pred_test = self.feed_forward(X_test)
            cost_ = self.cost_func(y_test,y_pred_test,deriv=False)
            error_test.append(cost_)
            """plt.plot(range(len(error)),error,label='train')
            plt.plot(range(len(error_test)),error_test,label='test')
            plt.title('Cost as a function of epochs (batch no. {})'.format(i))
            plt.legend()
            plt.show()"""

        return error, error_test


def sklearn_NN(X, y, X_val, y_val, lr):   #scikit Learns algorithm for NN on classification
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(activation='tanh',solver='sgd', alpha=lr, hidden_layer_sizes=(50,50,10), random_state=1,max_iter=1000)
    clf.fit(X, y)
    y_pred=clf.predict(X)
    y_pred_val = clf.predict(X_val)
    print("test")

    return y_pred,y_pred_val


def sklearn_NN_regressor(X, y, X_val, y_val, lr): #scikit Learns algorithm for NN on regression

    from sklearn.neural_network import MLPRegressor
    clf = MLPRegressor(activation='relu',solver='sgd', alpha=lr, hidden_layer_sizes=(50,50,10))
    clf.fit(X, y)
    y_pred=clf.predict(X)
    y_pred_val = clf.predict(X_val)

    return y_pred_val, y_pred
