

from Class import Regression, Error, Resample#, Error_Analysis, Resampling
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

from sklearn.model_selection import KFold

"""
Functions used for plotting. First Franke function,plot function for franke is defined. Various functions which gets data.
"""


def Franke_Func(x,y, n=20):   #f(x,y) #France function
    term1 = ( 0.75 * np.exp( -((9*x - 2)**2 / 4)  - ((9*y-2)**2 / 4)) )
    term2 = ( 0.75 * np.exp( -((9*x+1)**2 / 49) - ((9*y+1)**2 / 10 )) )
    term3 = ( 0.5 * np.exp( -((9*x-7)**2 / 4 ) - ((9*y-3)**2 / 4)) )
    term4 = -( 0.2 * np.exp( -(9*x-4)**2 - (9*y-7)**2 ) )

    return (term1 + term2 + term3 + term4)


def plot(x=np.arange(x=np.arange(0,1,0.05),y=np.arange(0,1,0.05,z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.scatter(x,y,z)
    x,y = np.meshgrid(x,y)
    surf = ax.plot_surface(x,y,z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    plt.xlabel('x')
    plt.ylabel('y')

    fig.colorbar(surf, shrink = 0.5, aspect=5, label='z')
    plt.show()



def design_matrix(x, y, deg): #copied from piaza

    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    p = int((deg + 1)*(deg + 2)/2)
    X = np.ones((N,p))

    for i in range(1, deg + 1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = x**(i-k) * y**k
    return X 


#No resample, predicts errors, and plots MSE and R2 as a function of polynomial degree
def getData_noRes(method, x,y,z,max_degree,Print_errors=True,plot_err=True, plot_BiVar=True):
    degrees = np.arange(1,max_degree+1)
    MSE = np.zeros(max_degree)
    R2 = np.zeros(max_degree)
    bias = np.zeros(max_degree)
    Var = np.zeros(max_degree)

    i = 0
    beta_interval = []
    for j in degrees:
        X = design_matrix(x,y,j)
        if method == 'OLS':
            mtd = Regression('OLS',X, z)
            z_pred = mtd.predict(X)
        elif method == 'Ridge':
            mtd = Regression('Ridge',X, z)
            z_pred = mtd.predict(X)
        elif method == 'Lasso':
            mtd = Regression('Lasso', X, z)
            z_pred = mtd.predict(X)


        MSE[i]=Error(z, z_pred).MSE()
        R2[i] = Error(z, z_pred).R2()
        bias[i]=Error(z, z_pred).bias()
        Var[i] = Error(z, z_pred).Var()
        i += 1


    if Print_errors==True:
        for k in range(len(degrees)):
            print("Degree: ", k)
            print("MSE: ", MSE[k])
            print("R2: ", R2[k])
            print("Bias: ", bias[k])
            print("Variance: ", Var[k])

    if plot_err==True:
        figure = plt.figure() #, ax1, ax2 = plt.subplots()
        figure.suptitle(r'R$^2$ and MSE of polydegree for Lasso ($\lambda=10^{-9}$) - no resampling')
        plt.subplot(2,1,1)
        plt.plot(degrees, MSE)
        plt.ylabel('MSE')
        #plt.xlabel('Polydegree')
        plt.subplot(2,1,2)
        plt.plot(degrees, R2)
        plt.ylabel(r'R$^2$')
        plt.xlabel('Polydegree')
        #plt.savefig('Ridge_noRes_MSE_R2_lamb_1e-5', dpi=300)
        plt.show()
    if plot_BiVar ==True:
        plt.plot(degrees, bias)
        plt.plot(degrees, Var)
        plt.legend([r'Bias$^2$', 'Variance'], loc='best')
        plt.title('Bias variance tradeoff with no resampling')
        plt.xlabel('polydegree')
        plt.ylabel('Error')
        plt.show()


    return MSE, R2, bias, Var #Get data from regression class only


#Bootstrap resample, predicts errors, and plots MSE as a function of polynomial degree, and bias variance tradeoff
def getData_Res_bootstrap(method,n_bootstraps, x,y,z,max_degree, Print_Val=True, plot_err=True, plot_BiVar=True):
    degrees = np.arange(1,max_degree+1)
    MSE_test = np.zeros(max_degree)
    MSE_train = np.zeros(max_degree)
    R2 = np.zeros(max_degree)
    bias = np.zeros(max_degree)
    Var = np.zeros(max_degree)



    i = 0
    for j in degrees:
        X = design_matrix(x,y,j)
        if method == 'OLS':
            res = Resample('OLS', X, x, y, z)
            MSE_test[i], MSE_train[i], bias[i], Var[i] = res.Bootstrap(n_bootstraps)
        elif method == 'Ridge':
            res = Resample('Ridge', X, x, y, z)
            #mtd = Regression('Ridge',X, z)
            MSE_test[i], MSE_train[i], bias[i], Var[i] = res.Bootstrap(n_bootstraps)
            #print(bias)
        elif method == 'Lasso':
            res = Resample('Lasso', X, x, y, z)
            #mtd = Regression('Lasso', X, z)
            MSE_test[i], MSE_train[i],bias[i], Var[i] = res.Bootstrap(n_bootstraps)



        i += 1
    if Print_Val == True:
        for k in range(len(degrees)):
            print("Degree: ", k)
            print("MSE test: ", MSE_test[k])
            print("MSE train: ", MSE_train[k])
            print("Bias: ", bias[k])
            print("Variance: ", Var[k])

    if plot_err==True:
        figure = plt.figure() #, ax1, ax2 = plt.subplots()
        plt.plot(degrees, MSE_test)
        plt.plot(degrees, MSE_train)
        plt.ylabel('MSE')
        plt.xlabel('Polydegree')
        plt.title('MSE of polydegree for Ridge ($\lambda=10^{-9}$) - Resampling')
        plt.legend(['MSE test', 'MSE train'], loc='best')
        plt.savefig('OLS_ResBootstrap_MSE_TrainTest', dpi=300)
        plt.show()

    if plot_BiVar ==True:
        plt.plot(degrees, bias)
        plt.plot(degrees, Var)
        plt.legend([r'Bias$^2$', 'Variance'], loc='best')
        plt.title('Bias variance tradeoff with resampling for Lasso ($\lambda=10^{-5}$)')
        plt.xlabel('polydegree')
        plt.ylabel('Error')
        #plt.savefig('OLS_biasVar_Bootstrap', dpi=300)
        plt.show()

#Resample K fold cross validation. Plots and print the MSE as a function of polynomial degree
def getData_Res_Kfold(method, k, x,y,z,max_degree, plot_MSE=True, Print_MSE=True):  #get data from KFold
    degrees = np.arange(1,max_degree+1)
    error = np.zeros(max_degree)
    MSE_train = np.zeros(max_degree)
    MSE_test = np.zeros(max_degree)
    variance = np.zeros(max_degree)
    R2_test= np.zeros(max_degree)
    R2_train= np.zeros(max_degree)
    bias = np.zeros(max_degree)
    Var = np.zeros(max_degree)

    i = 0
    for j in degrees:
        X = design_matrix(x,y,j)
        if method == 'OLS':
            res = Resample('OLS', X, x, y, z)
            R2_test[i], R2_train[i], MSE_test[i], MSE_train[i] = res.KFold_CrossVal(k)
        elif method == 'Ridge':
            res = Resample('Ridge', X, x,y,z)
            R2_test[i], R2_train[i], MSE_test[i], MSE_train[i] = res.KFold_CrossVal(k)
        elif method == 'Lasso':
            res = Resample('Lasso', X, x,y,z)
            R2_test[i], R2_train[i], MSE_test[i], MSE_train[i] = res.KFold_CrossVal(k)
        i += 1

    if plot_MSE == True:
        figure = plt.figure()
        plt.plot(degrees, MSE_test)
        plt.plot(degrees, MSE_train)
        plt.title('MSE of polydegree for Lasso ($\lambda=10^{-9}$) - with resampling')
        plt.ylabel('MSE')
        plt.legend(['MSE test data', 'MSE train data', ], loc='best')
        plt.xlabel('Polydegree')
        plt.show()




        print(MSE_test)
    if Print_MSE == True:
        for k in range(len(degrees)):
            print("Degrees :", k )
            print("MSE test data: ", MSE_test[k])
            print("MSE train data: ", MSE_train[k])
            print("R2 test data: ", R2_test[k])
            print("R2 train data: ", R2_train[k])



#An attempt to plot MSE and bias-variance tradeoff with one polynomial degree but different values of lambda. No success.
def getData_various_lamb_res(model, x,y,z, polydegree):
    nlambdas = 500
    lambdas = np.logspace(-3, 5, nlambdas)
    X = design_matrix(x,y,polydegree)
    k= 5
    estimated_MSE= np.zeros(nlambdas)
    kf = KFold(n_splits=k, shuffle=True)
    for lmb in lambdas:
        if model == 'Ridge':
            i=0
            for train_index, test_index in kf.split(z):
                z_train, z_test = z[train_index], z[test_index]
                X_train, X_test = X[train_index], X[test_index]
                beta = Regression('Ridge', X_train, z_train).Ridge(lmb, get_beta=True)
                z_pred = X_test @ beta
                estimated_MSE[i]=Error(z_test, z_pred).MSE()
                #if np.shape(z_pred)==np.shape(z_test):
                #    print("yeah")
                #estimated_MSE[i] = z_test-z_pred
                i += 1
    #print(np.shape(estimated_MSE))
    #print(estimated_MSE)
    #plt.plot(np.log10(lambdas), estimated_MSE)
 




#An attempt to get the beta confidence interval. No success
def getData_betainterval(x,y,z,degree):

    for j in np.arange(1,degree+1):

        X= design_matrix(x,y,j)
        betas = Regression('OLS', X,z).OLS(get_beta=True)
        print(betas[1])
        print(np.mean(betas[1]) )
        #var = np.mean(betas**2)-(np.mean(betas))**2
        #print(var)
        #print(np.variance(betas))

        #print(betas)
        #beta = np.mean(np.abs(betas))
        #beta = np.mean(betas)
        #beta_var = (np.std(betas))# * 1.96
        #print(beta_var)
        #print(beta)
        #print("degree: ", j, beta, beta+beta_var, beta-beta_var  )
        #print(beta_var)
        #print("degree: %d, beta: %.4f :  [%.4f , %.4f]" %(j,beta, (beta+beta_var), (beta-beta_var)), beta_var)

