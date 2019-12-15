
from sklearn.datasets import load_breast_cancer

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class ReadData:

    def __init__(self, data):
        self.data = data


    def get_data(self, return_names=False):
        X_pd = pd.DataFrame(self.data.data, columns=self.data.feature_names)
        y_pd = pd.Categorical.from_codes(self.data['target'], self.data['target_names'])  ## contain M and B
        y_pd = pd.get_dummies(y_pd)   ## contain M=0 and B=1.

        X = X_pd.to_numpy()[:,]
        y = y_pd.to_numpy().T[1,:]

        X_train, X_test, y_train, y_test = self.ScaleSplit_data(X,y)


        if return_names:
            feature_names = self.data.feature_names
            class_names = self.data.target_names
            return class_names, feature_names

        else:
            return X_train, X_test, y_train, y_test


    def ScaleSplit_data(self, X, y):
        MM_Sc = MinMaxScaler()

        trainingShare = 0.8
        seed = 1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-trainingShare, random_state=seed,shuffle=True)

        X_train = MM_Sc.fit_transform(X_train)
        X_test = MM_Sc.transform(X_test)


        return X_train, X_test, y_train, y_test

