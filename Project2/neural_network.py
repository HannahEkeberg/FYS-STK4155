import numpy as np
#print("numpy version: ", np.version.version)

class NeuralNetwork:
    def __init__(self, X, y, n_hidden_neurons=50 , n_categories=10 , epochs=100 , batch_size=100 , eta=0.1 , lmbd=0.0):

        self.X = X
        self.y = y
        self.X_full = self.X
        self.y_full = self.y

        self.n_inputs         = X.shape[0]  #X-rows
        self.n_features       = X.shape[1]  #X-columns
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories     = n_categories

        self.epochs           = epochs
        self.batch_size       = batch_size
        self.iterations       = self.n_inputs // self.batch_size #Floor division
        self.eta              = eta
        self.lmbd             = lmbd

        self.create_biases_and_weights() #gives the initial bias and weigths.
        #self.feed_forward()
        #print(self.hidden_weights)
        #print("***")
        #self.backpropagation()
        #print(self.hidden_weights)

    def sigmoid(self,x):
        return(1/(1 + np.exp(-x)))

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias    = np.zeros(self.n_hidden_neurons) + 0.01
        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias    = np.zeros(self.n_categories) + 0.01


    def feed_forward(self):
        #feed_forward training

        #self.z_h = np.matmul(self.X, self.hidden_weights) + self.hidden_bias
        self.z_h = (self.X @ self.hidden_weights) + self.hidden_bias

        self.a_h = self.sigmoid(self.z_h)


        #self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias
        self.z_o = (self.a_h @ self.output_weights) + self.output_bias



        exp_term = np.exp(self.z_o)
        self.probabilities = exp_term /np.sum(exp_term, axis=1, keepdims=True)  #divide on the second axis -> len=50


    def feed_forward_output(self, X):
        #feed_forward for output
        #z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        z_h = (X @ self.hidden_weights) + self.hidden_bias


        a_h = self.sigmoid(z_h)

        #z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        z_o = (a_h @ self.output_weights) + self.output_bias

        exp_term = np.exp(z_o)
        probabilities = exp_term/ np.sum(exp_term, axis=1, keepdims=True)
        return probabilities

    def backpropagation(self):

        error_output = self.probabilities - self.y[0]  #cost function
        error_hidden = (error_output @ self.output_weights.T) *self.a_h * (1-self.a_h)


        self.output_weights_gradient = (self.a_h.T @ error_output)
        self.output_bias_gradient    = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = (self.X.T @ error_hidden)
        self.hidden_bias_gradient    = np.sum(error_hidden, axis=0)


        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias    -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias    -= self.eta * self.hidden_bias_gradient

    def predict(self, X):
        probabilities = self.feed_forward_output(X)
        return np.argmax(probabilities, axis = 1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_output(X)
        return probabilities

    def train(self):
        data_ind = np.arange(self.n_inputs)
        print("hellu")
        for i in range(self.epochs):
            for j in range(self.iterations):
                #pick datapoints with replacements
                chosen_datapoints = np.random.choice(data_ind, size=self.batch_size, replace=False)

                #minibatch training data
                self.X_full = self.X[chosen_datapoints]
                self.y_full = self.y[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()
