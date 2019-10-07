from getData import *
import random


#Defining x, y and z
n=1000
random.seed(2018)
x = np.random.random(n)
y = np.random.random(n)
random.seed(2018)
noise = 0.01*np.random.randn(n)   #0.1*np.random.randn(20,1)  #(mu=0, sigma^2=1)
z = Franke_Func(x,y) + noise #f(x,y)+epsilon


###Plot of MSE and R2 for OLS ridge and Lasso with no resampling
#getData_noRes('Lasso', x,y,z,5, Print_errors=True, plot_err=True, plot_BiVar=False)



### Get beta confidence interval
#getData_betainterval(x,y,z,5)


### Bias variance tradeoff and MSE with train and test for bootstrap
#getData_Res_bootstrap('Lasso', 100, x,y,z,max_degree=20,Print_Val=True,plot_err=False, plot_BiVar=True)


### MSE with train and test data for K Fold
#getData_Res_Kfold('Lasso', 5, x,y,z,15, plot_MSE=True, Print_MSE=True)


### Bias and variance, MSE of Ridge and Lasso for various values of lambda
#getData_various_lamb_res('OLS', x,y,z, 6)
