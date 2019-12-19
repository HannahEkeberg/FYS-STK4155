
from sklearn.datasets import load_breast_cancer

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()


class ReadData:

    def __init__(self, data):
        self.data = data



    def get_data(self, correlation_drop=False, return_names=False):
        # Dataframe of features
        X_pd = pd.DataFrame(self.data.data, columns=self.data.feature_names)

        # Dataframe of target
        y_pd = pd.Categorical.from_codes(self.data['target'], self.data['target_names'])  ## contain M and B

        # Makes dataframe with columns: index;  M=1, B=0, M=0, B=1
        # M = malignant, B = benign
        y_pd = pd.get_dummies(y_pd)

        # Names of the feature and target names
        feature_names = self.data.feature_names
        class_names = self.data.target_names

        # drop features with higher correlation than 95%
        if correlation_drop:
            X_pd = self.correlation_features(X_pd)
            feature_names = X_pd.columns

        # Convert to numpy arrays, where y represents M=0, and B=1
        X = X_pd.to_numpy()[:,]
        y = y_pd.to_numpy().T[1,:]

        ## Scaling and splitting
        X_train, X_test, y_train, y_test = self.ScaleSplit_data(X,y)


        if return_names:
            return class_names, feature_names

        else:
            return X_train, X_test, y_train, y_test


    def ScaleSplit_data(self, X, y):
        #Sc = StandardScaler()
        #MnMxSc = MinMaxScaler()
        MnMxSc = StandardScaler()

        # Splitting data into train and test
        trainingShare = 0.8
        seed = 1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-trainingShare, random_state=seed,shuffle=True)

        # Scaling of data using StandardScaler
        X_train = MnMxSc.fit_transform(X_train)
        X_test = MnMxSc.transform(X_test)
        return X_train, X_test, y_train, y_test

    def correlation_features(self, X, makePlot=False):
        corr_matrix = X.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Find index of feature columns with correlation greater than 0.95
        drop_columns = [column for column in upper.columns if any(upper[column] > 0.95)]


        if makePlot:
            ax = sns.heatmap(corr_matrix)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize = 8)
            ax.set_yticklabels(ax.get_yticklabels(), rotation = 45, fontsize = 8)
            plt.savefig('Figures/correlation_heatmap.png', dpi=300)
            plt.show()
            print(drop_columns)
            print(upper.columns)

        # Dropping features with higher correlation than 95%
        X = X.drop(X[drop_columns], axis=1)


        return X
