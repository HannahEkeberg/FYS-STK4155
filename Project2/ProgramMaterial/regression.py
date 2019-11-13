import numpy as np
from sklearn.linear_model import LinearRegression as OLS, Ridge, Lasso

"""
Default data set: Franke Function.
Change the cost function in order to perform a regression.
We need to find
"""

class Regression:

    def __init__(self, model, lamb, X, z):
        self.model = model
        self.lamb = lamb
        self.X = X
        self.z = z

    def OLS(self):
        self.beta_OLS = np.linalg.pinv(self.X.T.dot(self.X)) @ self.X.T.dot(self.z)

    def Ridge(self):
        n,p=np.shape(self.X)
        I_lambda = np.identity(p, dtype=None)*self.lamb
        self.beta_ridge = np.linalg.inv(self.X.T.dot(self.X) + I_lambda) @ (self.X.T.dot(self.z))

    def Lasso(self):

        self.beta_Lasso = Lasso(alpha=self.lamb, fit_intercept=False,
                normalize=False, max_iter=1000).fit(self.X, self.z).coef_

    def predict(self, X):
        if self.model == 'OLS':
            self.OLS()
            return X @ self.beta_OLS
        elif self.model == 'Ridge':
            self.Ridge()
            #print(self.beta_ridge)
            return X @ self.beta_ridge
        elif self.model == 'Lasso':
            self.Lasso()
            return X @ self.beta_Lasso

    def sklearn_reg(self, X):
        if self.model == 'OLS':
            clf = OLS()
            clf.fit(self.X, self.z)
            y_pred = clf.predict(X)

        elif self.model == 'Ridge':
            clf = Ridge(alpha=self.lamb)
            y_pred = clf.predict(X)

        elif self.model == 'Lasso':
            clf = Lasso(alpha=self.lamb, max_iter=10000, normalize=False, tol=0.0001)
            clf.fit(self.X, self.z)
            y_pred = clf.predict(X)

        return y_pred
