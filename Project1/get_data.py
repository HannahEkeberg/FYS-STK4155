

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





def Franke_Func(x,y, n=20):   #f(x,y) #France function
    term1 = ( 0.75 * np.exp( -((9*x - 2)**2 / 4)  - ((9*y-2)**2 / 4)) )
    term2 = ( 0.75 * np.exp( -((9*x+1)**2 / 49) - ((9*y+1)**2 / 10 )) )
    term3 = ( 0.5 * np.exp( -((9*x-7)**2 / 4 ) - ((9*y-3)**2 / 4)) )
    term4 = -( 0.2 * np.exp( -(9*x-4)**2 - (9*y-7)**2 ) )
    #noise = np.random.normal(0, 0, n)
    return (term1 + term2 + term3 + term4)# + noise)

#Defining x, y and z
#x = np.arange(0, 1, 0.05)
#y = np.arange(0, 1, 0.05)
#x,y = np.meshgrid(x,y)
x = np.linspace(0,1,20)
y = np.linspace(0,1,20)
noise = 0.1*np.random.randn(20,1)  #(mu=0, sigma^2=1)
z = Franke_Func(x,y) + noise #f(x,y)+epsilon

z_flat = np.ravel(z)   #1D array 20x20 long


"""
##making design matrix
n = len(x); p = 6
X = np.zeros((n,p))
X[:,0] = 1
X[:,1] = x
X[:,2] = y
X[:,3] = (x**2)
X[:,4] = (y**2)
X[:,5] = (x*y)
"""
def plot(x,y,z, z_pred):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x,y = np.meshgrid(x,y)
    surf = ax.plot_surface(x,y,z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    plt.xlabel('x')
    plt.ylabel('y')
    #plt.zlabel('z')

    fig.colorbar(surf, shrink = 0.5, aspect=5, label='z')
    plt.show()


def plot_model(data, model):
    pass

def design_matrix(x, y, deg):

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

    return X #Not original

#def design_matrix(x,y, degree):
#    X_ = np.c_[x,y]
#    poly = PolynomialFeatures(degree)
#    X = poly.fit_transform(X_)
#    return X


###a) OLS

def getData_noRes(method, x,y,z,max_degree,Print_errors=True,plot_err=True, plot_BiVar=True):
    degrees = np.arange(1,max_degree+1)
    MSE = np.zeros(max_degree)
    R2 = np.zeros(max_degree)
    bias = np.zeros(max_degree)
    Var = np.zeros(max_degree)
    #lamb = [0.0001, 0.001, 0.01, 0.1]
    lamb = 0.001

    i = 0
    beta_interval = []
    for j in degrees:
        X = design_matrix(x,y,j)
        if method == 'OLS':
            mtd = Regression('OLS',X, z)
            z_pred = mtd.predict(X)
            #beta = (mtd.OLS(get_beta=True))
            #print(np.shape(beta))
            #Var[i] = Error(z, z_pred).Var()
            #print("Beta interval degree:", j, "[%.3f ,%.3f ]" %(beta+1.96*np.sqrt(Var[i]**2), (beta-1.96*np.sqrt(Var[i]**2)  )))
            #beta_interval.append(beta+1.96*np.sqrt(Var[i]**2))
            #(beta+1.96*np.sqrt(Var[i]**2).append(beta_interval)
            #(beta-1.96*np.sqrt(Var[i])**2).append(beta_interval)
        elif method == 'Ridge':
            mtd = Regression('Ridge',X, z)
            z_pred = mtd.predict(X)
        elif method == 'Lasso':
            mtd = Regression('Lasso', X, z)
            z_pred = mtd.predict(X)


        print(beta_interval)
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
        figure.suptitle(r'R$^2$ and MSE of polydegree for OLS - no resampling')
        plt.subplot(2,1,1)
        plt.plot(degrees, MSE)
        plt.ylabel('MSE')
        #plt.xlabel('Polydegree')
        plt.subplot(2,1,2)
        plt.plot(degrees, R2)
        plt.ylabel(r'R$^2$')
        plt.xlabel('Polydegree')
        plt.savefig('OLS_noRes_MSE_R2', dpi=300)
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



#getData_noRes('OLS', x,y,z,20, Print_errors=True, plot_err=True, plot_BiVar=True)

X=design_matrix(x,y,deg=6)
reg = Regression('OLS', X, z)
beta = reg.OLS(get_beta=True)


def getData_Res_bootstrap(method,n_bootstraps, x,y,z,max_degree, print=True, plot_err=True, plot_BiVar=True):
    degrees = np.arange(1,max_degree+1)
    MSE = np.zeros(max_degree)
    R2 = np.zeros(max_degree)
    bias = np.zeros(max_degree)
    Var = np.zeros(max_degree)



    i = 0
    for j in degrees:
        X = design_matrix(x,y,j)
        if method == 'OLS':
            res = Resample('OLS', X, x, y, z)
            MSE[i], R2[i],bias[i], Var[i] = res.Bootstrap(n_bootstraps)
        elif method == 'Ridge':
            mtd = Regression('Ridge',X, z)
            MSE[i], R2[i],bias[i], Var[i] = res.Bootstrap(n_bootstraps)
        elif method == 'Lasso':
            mtd = Regression('Lasso', X, z)
            MSE[i], R2[i],bias[i], Var[i] = res.Bootstrap(n_bootstraps)



        #MSE[i]=Error(z, z_pred).MSE()
        #R2[i] = Error(z, z_pred).R2()
        #bias[i]=Error(z, z_pred).bias()
        #Var[i] = Error(z, z_pred).Var()
        i += 1
    if print == True:
        for k in range(len(degrees)):
            print("Degree: ", k)
            print("MSE: ", MSE[k])
            print("R2: ", R2[k])
            print("Bias: ", bias[k])
            print("Variance: ", Var[k])

    if plot_err==True:
        figure = plt.figure() #, ax1, ax2 = plt.subplots()
        figure.suptitle(r'R$^2$ and MSE of polydegree for OLS - no resampling')
        plt.subplot(2,1,1)
        plt.plot(degrees, MSE)
        plt.ylabel('MSE')
        #plt.xlabel('Polydegree')
        plt.subplot(2,1,2)
        plt.plot(degrees, R2)
        plt.ylabel(r'R$^2$')
        plt.xlabel('Polydegree')
        plt.savefig('OLS_noRes_MSE_R2', dpi=300)
        plt.show()

    if plot_BiVar ==True:
        plt.plot(degrees, bias)
        plt.plot(degrees, Var)
        plt.legend([r'Bias$^2$', 'Variance'], loc='best')
        plt.title('Bias variance tradeoff with no resampling')
        plt.xlabel('polydegree')
        plt.ylabel('Error')
        plt.show()


    return MSE, R2, bias, Var #get data from bootstrap
#getData_Res_bootstrap('OLS', 100, x,y,z,20,print=False,plot_err=False, plot_BiVar=True)


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
            error[i], bias[i], variance[i], R2_test[i], R2_train[i], MSE_test[i], MSE_train[i] = res.KFold_CrossVal(k)
    if plot_MSE == True:
        figure = plt.figure() #, ax1, ax2 = plt.subplots()
        figure.suptitle(r'R$^2$ and MSE of polydegree for OLS - with resampling')
        plt.subplot(2,1,1)
        plt.plot(degrees, MSE_test)
        plt.plot(degrees, MSE_train)
        plt.ylabel('MSE')
        plt.legend(['MSE test data', 'MSE train data', ], loc='best')
        #plt.xlabel('Polydegree')
        plt.subplot(2,1,2)
        plt.plot(degrees, R2)
        plt.ylabel(r'R$^2$')
        plt.xlabel('Polydegree')
        plt.legend([r'R$^2$ test data', 'R$^2$ train data' ])
        plt.savefig('OLS_Res_MSE_R2', dpi=300)
        plt.show()
        #plt.title('MSE of the train and test data as a function of polynomial degree')



    if Print_MSE == True:
        for i in degrees:
            print("Degrees :", i )
            print("MSE test data: ", MSE_test[i])
            print("MSE train data: ", MSE_train[i])


getData_Res_Kfold('Ridge', 5, x,y,z,5, plot_MSE=True, Print_MSE=False)


def getData_various_lamb_res(method, x,y,z, polydegree):

    lamb = np.array((0.0001, 0.001, 0.01, 0.1, 1.0))
    X = design_matrix(x,y, degree=polydegree)
    z_train, z_test, X_train, X_test = train_test_split(z,X,test_size=0.2)

    ###doing Kfold manually
    for lmb in lamb:
        beta = Regression('Ridge', X_train,z_train).Ridge(lmb, get_beta=True)
        z_test = X_test @ beta








#getData_various_lamb_res('Ridge', x,y,z, 6)






