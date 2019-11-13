

from logistic_regression import LogisticRegression_class
from creditcard_data import *   # Arranging the creditcard data. Returns X train, X test, y train, y test
from scores import *            # Accuracy score, confusion matrix, MSE, R2
from cost_act_funcs import *    # Neg log likelihood, MSE, Sigmoid, tanh, Relu
from neural_network import *    # Analytical neural network class, Sklearn classification MLP, Sklearn regressor MLP
from regression import *        # OLS, Ridge and Lasso (analytical), Sklearn OLS, Ridge and Lasso
from Franke_function import  *  # Design matrix, Franke function

import matplotlib.pyplot as plt

"""
Logistic regression output is in Logistic_Reg. If costfunction plot: MakePlot=True, if confusion matrix: CM=True


"""


def Logistic_Reg(lr, n_epochs, batch_size, MakePlot=False, CM = False):
    X_train, X_test, y_train, y_test = balanced_data()
    LogReg = LogisticRegression_class(lr, batch_size, n_epochs)
    error_train, error_val, y_pred, y_val = LogReg.gradient_descent(X_train, y_train, X_test, y_test)
    y_skl_val = LogReg.predict_sklearn(X_train, X_test, y_train, y_test)
    if MakePlot:
        plt.plot(range(n_epochs), error_train,label='Train')
        plt.plot(range(n_epochs),error_val,label='Val')
        #ax = plt.gca()
        #ax.set_facecolor("white")
        plt.legend()
        #plt.grid('off')
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.title('Cost as a function of epochs (learning rate: {}, batchsize: {})'.format(lr, batch_size))
        plt.savefig('Figures/cost_func_{}.png'.format(lr), dpi=300)
        plt.show()

    ## Accuracy scores:
    score_train = accuracy_score(y_train, y_pred)
    score_val = accuracy_score(y_test, y_val)
    score_sk_val = accuracy_score(y_test, y_skl_val)
    print("train: ",score_train)
    print("val: ",score_val)
    print("sklearn: ", score_sk_val)


    if CM:
        accuracy_matrix(y_train, y_pred, 'Confusion Matrix on Train Data For Log. Regression')
        accuracy_matrix(y_test, y_val, 'Confusion Matrix on Val data for log. regression')
        accuracy_matrix(y_test, y_skl_val, 'Confusion Matrix (Sklearn) for log. regression')



#Logistic_Reg(lr=0.1, n_epochs=120, batch_size=128, CM=True)



def NN(lr, batch_size, epochs, layers=[50,50,10], act_func=tanh, MakePlot=False, CM=False): #CM: confusion matrix plot
    X_train, X_test, y_train, y_test = balanced_data()

    ##Analytical
    n_inputs = X_train.shape[1]
    n_outputs = 1
    #activation_function = [tanh,tanh,tanh,sigmoid_function]  #activation function for hidden layers and outputlayer[-1] (always sigmoid in classification binary problems since want outcome specifically 0 or 1)
    activation_function = [act_func, act_func, act_func, sigmoid_function]
    cost_function = neg_log
    NN = NeuralNetwork(layers, n_inputs, n_outputs,activation_function, cost_function)
    error_train, error_val = NN.train(X_train,y_train,X_test,y_test,lr,batch_size,epochs)

    y_pred_val = NN.feed_forward(X_test)
    y_pred_train =NN.feed_forward(X_train)

    tol = 0.5
    y_pred_val[y_pred_val >= tol] = 1
    y_pred_val[y_pred_val < tol] = 0
    y_pred_train[y_pred_train >= tol] = 1
    y_pred_train[y_pred_train < tol] = 0



    score_train = accuracy_function(y_train, y_pred_train)
    print("train score: ", score_train)
    score_test = accuracy_function(y_test, y_pred_val)
    print("test score: ", score_test)



    if MakePlot:
        plt.plot(range(epochs), error_train, label='train')
        plt.plot(range(epochs), error_val, label='val')
        plt.legend()
        plt.title('Cost as a function of epochs with learning rate {} for {} '.format(lr, activation_function[0].__name__))
        plt.xlabel('epochs')
        plt.ylabel('Cost')
        plt.savefig('Figures/NN_{}_classification_cost.png'.format(activation_function[0].__name__), dpi=300)
        plt.legend()
        plt.show()


    #SkLearn
    y_pred, y_val = sklearn_NN(X_train, y_train, X_test, y_test, lr)
    score_train = accuracy_function(y_train, y_pred)
    print("train: ",score_train)
    score_test = accuracy_function(y_test, y_val)
    print("val: ",score_test)


    if CM:
        accuracy_matrix(y_train, y_pred, 'NN training data - Sklearn')
        accuracy_matrix(y_test, y_val, 'NN validation data - Sklearn')
        accuracy_matrix(y_train, y_pred_train, 'NN training data')
        accuracy_matrix(y_test, y_pred_val, 'NN validation data')

#NN(lr=1e-3, epochs=100, batch_size=128, act_func=tanh)


def linear_regression(method, n=1000):
    np.random.seed(2018)
    x = np.random.random(n)
    y = np.random.random(n)
    noise = 0.1*np.random.randn(n)
    z = Franke_Func(x,y) + noise
    X = design_matrix(x,y, 10) #degree 10 as default
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    lambda_ = [1e-10,1e-5, 1e-4,1e-3, 1e-2, 1e-1, 1e0, 1e1]
    print("**",method,"**")

    #from sklearn.metrics import mean_squared_error, r2_score
    #print(X_train.shape, X_test.shape, z_train.shape, z_test.shape)
    if method == 'OLS':
        z_pred_train = Regression('OLS', 0, X_train, z_train ).predict(X_train)
        z_pred_val = Regression('OLS', 0, X_train, z_train).predict(X_test)
        z_pred_train_sklearn = Regression('OLS', 0, X_train, z_train).sklearn_reg(X_train)
        z_pred_val_sklearn = Regression('OLS', 0, X_train, z_train).sklearn_reg(X_test)
        MSE_train = MSE_func(z_train, z_pred_train)
        MSE_val = MSE_func(z_test, z_pred_val)
        MSE_sklearn_train = MSE_func(z_train, z_pred_train_sklearn)
        MSE_sklearn_val = MSE_func(z_test, z_pred_val_sklearn)
        R2_train = R2_func(z_train, z_pred_train)
        R2_val = R2_func(z_test, z_pred_val)
        R2_sklearn_train = R2_func(z_train, z_pred_train_sklearn)
        R2_sklearn_val = R2_func(z_test, z_pred_val_sklearn)
        #print("MSE:")
        print("MSE train", MSE_train, "MSE val", MSE_val)
        print("MSE train sklean", MSE_sklearn_train, "MSE val sklearn", MSE_sklearn_val)
        print("R2 train", R2_train, "R2 val", R2_val)
        print("R2 train sklearn", R2_sklearn_train, "R2 val sklearn", R2_sklearn_val)


    elif method == 'Ridge':
        MSE_train = []
        MSE_val = []
        MSE_train_sklearn = []
        MSE_val_sklearn = []

        R2_train = []
        R2_val = []
        R2_train_sklearn = []
        R2_val_sklearn = []

        for i in range(len(lambda_)):
            lmbd = lambda_[i]
            z_pred_train = Regression('Ridge', lmbd, X_train, z_train ).predict(X_train)
            z_pred_val = Regression('Ridge', 0, X_train, z_train).predict(X_test)
            z_pred_train_sklearn = Regression('Ridge', 0, X_train, z_train).sklearn_reg(X_train)
            z_pred_val_sklearn = Regression('Ridge', 0, X_train, z_train).sklearn_reg(X_test)

            MSE_train.append(MSE_func(z_train, z_pred_train))
            MSE_val.append(MSE_func(z_test, z_pred_val))
            MSE_train_sklearn.append(MSE_func(z_train, z_pred_train_sklearn))
            MSE_val_sklearn.append(MSE_func(z_test, z_pred_val_sklearn))

            R2_train.append(R2_func(z_train, z_pred_train))
            R2_val.append(R2_func(z_test, z_pred_val))
            R2_train_sklearn.append(R2_func(z_train, z_pred_train_sklearn))
            R2_val_sklearn.append(R2_func(z_test, z_pred_val_sklearn))

        for i in range(len(lambda_)):
            print("lambda: ", lambda_[i])
            print("MSE train:", MSE_train[i], "MSE val", MSE_val[i])
            print("MSE train sklearn:", MSE_train_sklearn[i], "MSE val sklearn", MSE_val_sklearn[i])
            print("R2 train: ", R2_train[i], "R2 val", R2_val[i])
            print("R2 train sklearn: ", R2_train_sklearn[i], "R2 val sklearn", R2_val_sklearn[i])


    elif method == 'Lasso':
        MSE_train = []
        MSE_val = []
        MSE_train_sklearn = []
        MSE_val_sklearn = []

        R2_train = []
        R2_val = []
        R2_train_sklearn = []
        R2_val_sklearn = []

        for i in range(len(lambda_)):
            lmbd = lambda_[i]
            z_pred_train = Regression('Lasso', lmbd, X_train, z_train ).predict(X_train)
            z_pred_val = Regression('Lasso', 0, X_train, z_train).predict(X_test)
            z_pred_train_sklearn = Regression('Lasso', 0, X_train, z_train).sklearn_reg(X_train)
            z_pred_val_sklearn = Regression('Lasso', 0, X_train, z_train).sklearn_reg(X_test)

            MSE_train.append(MSE_func(z_train, z_pred_train))
            MSE_val.append(MSE_func(z_test, z_pred_val))
            MSE_train_sklearn.append(MSE_func(z_train, z_pred_train_sklearn))
            MSE_val_sklearn.append(MSE_func(z_test, z_pred_val_sklearn))

            R2_train.append(R2_func(z_train, z_pred_train))
            R2_val.append(R2_func(z_test, z_pred_val))
            R2_train_sklearn.append(R2_func(z_train, z_pred_train_sklearn))
            R2_val_sklearn.append(R2_func(z_test, z_pred_val_sklearn))

        for i in range(len(lambda_)):
            print("lambda: ", lambda_[i])
            print("MSE train:", MSE_train[i], "MSE val", MSE_val[i])
            print("MSE train sklearn:", MSE_train_sklearn[i], "MSE val sklearn", MSE_val_sklearn[i])
            print("R2 train: ", R2_train[i], "R2 val", R2_val[i])
            print("R2 train sklearn: ", R2_train_sklearn[i], "R2 val sklearn", R2_val_sklearn[i])



linear_regression('Ridge', n=1000)
#MSE_OLS, MSE_Ridge, MSE_Lasso = linear_regression('OLS', n=1000)



def NN_regression(lr, batch_size, epochs, n, layers=[50,50,50]):
    x = np.random.random(n)
    y = np.random.random(n)
    np.random.seed(2018)
    noise = 0.1*np.random.randn(n)   #0.1*np.random.randn(20,1)  #(mu=0, sigma^2=1)
    z = Franke_Func(x,y) + noise #f(x,y)+epsilon
    X = design_matrix(x,y, 10)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
    n_inputs = X_train.shape[1]
    n_outputs = 1
    #print(type(n_inputs))
    activation_function = [ReLu,ReLu,ReLu,ReLu]   #setting relu as output layer activation function for
    cost_function = MSE

    NN_reg = NeuralNetwork(layers, n_inputs, n_outputs, activation_function, cost_function)

    error_train, error_val = NN_reg.train(X_train,z_train,X_test,z_test,lr,batch_size,epochs)

    z_pred_val = NN_reg.feed_forward(X_test)   #calling analytical network
    z_pred_train =NN_reg.feed_forward(X_train)

    z_pred_sklearn_val,z_pred_sklearn_train = sklearn_NN_regressor(X_train, z_train, X_test, z_test, lr) #scikitlearn NN


    z_pred_sklearn_train = np.ravel(z_pred_sklearn_train)
    z_pred_sklearn_val = np.ravel(z_pred_sklearn_val)


    MSE_train = MSE_func(z_train, z_pred_train)
    MSE_test = MSE_func(z_test, z_pred_val)
    MSE_train_sklearn = MSE_func(z_train, z_pred_sklearn_train)
    MSE_test_sklearn = MSE_func(z_test, z_pred_sklearn_val)
    R2_train = R2_func(z_train, z_pred_train)
    R2_test = R2_func(z_test, z_pred_val)
    R2_train_sklearn = R2_func(z_train, z_pred_sklearn_train)
    R2_test_sklearn = R2_func(z_test, z_pred_sklearn_val)


    print("MSE train data: ", MSE_train)
    print("MSE validation data: ", MSE_test)
    print("MSE train data sklearn: ", MSE_train_sklearn)
    print("MSE validation data sklearn: ", MSE_test_sklearn)
    print("R2 train data: ", R2_train)
    print("R2 validation data: ", R2_test)
    print("R2 train data sklearn: ", R2_train_sklearn)
    print("R2 validation data sklearn: ", R2_test_sklearn)


#NN_regression(1e-1, 128, 100, 1000)
