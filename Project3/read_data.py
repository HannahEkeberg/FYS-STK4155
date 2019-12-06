import pandas as pd
import os
import numpy as np
import matplotlib.pylab as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler #,OneHotEncoder
"""
#from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

from pydot import graph_from_dot_data
"""

def get_data(data, get_histogram=False):
    #BC = load_breast_cancer()

    X_pd = pd.DataFrame(data['data'], columns=data['feature_names'])
    y_pd = pd.Categorical.from_codes(data['target'], data['target_names'])  ## contain M and B

    ### gets 3 colums, 1: index, 2: M=1, B=0, 3: M=0, B=1.
    y_pd = pd.get_dummies(y_pd)   ## contain M=0 and B=1.


    ### Choose last column, so M=0, B=1
    y = y_pd.to_numpy().T[1,:]
    X = X_pd.to_numpy()[:,0:9]
    n = X.shape[0]
    #p = X.shape[1]

    no_B = np.count_nonzero(y)
    no_M = n - no_B  #n = no of rows
    names = ['Malignant={}'.format(no_M), 'Benign={}'.format(no_B)]

    if get_histogram:
        make_histogram(y, names, 'Proportion of benign and malignant tumors', 'histogram_prop')

    X_train, X_test, y_train, y_test = train_test_split_data(X,y)

    #Feature scaling:
    sc = StandardScaler()
    mm_sc = MinMaxScaler()
    #X_train = sc.fit_transform(X_train)    # StandardScaler
    #X_test = sc.transform(X_test)

    X_train = mm_sc.fit_transform(X_train) # MinMaxScaler
    X_test = mm_sc.transform(X_test)


    return X_train, X_test, y_train, y_test

def save_fig(fig_id):
    ### saving figure in directory Figures
    path = os.getcwd() + '/Figures/'
    string = path + fig_id + '.png'
    plt.savefig(path + fig_id + '.png', dpi=300)

def make_histogram(y, names, title_name, save_name):
    fig, ax = plt.subplots()
    ax.hist(y, align='left', bins=range(len(names)+1),rwidth=.9)

    xticks = [name for name in range(len(names))]
    xtick_labels = names
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    plt.title(title_name)
    save_fig(save_name)
    plt.show()



def train_test_split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    return X_train, X_test, y_train, y_test





