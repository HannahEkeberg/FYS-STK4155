from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import os

from sklearn.tree import export_graphviz
from IPython.display import Image
from pydot import graph_from_dot_data

#import matplotlib.pyplot as plt
from analyse import Analyse


"""
Class which contains SkLearn methods: DecisionTreeClassifier,
RandomForestClassifier, AdaBoostClassifier, and MLPClassifier.
Takes in splitted features and targets, and the column namesself.

In general, each method takes in various hyper parameters, and fits
method to training data.
Each method returns a predicted y value, y_pred_train predicted on training data, and
a y value, y_pred_val predicted on test data.

For DecisionTree, there is also included boolean variable to visaulize the tree,
set to False as default value.

"""

class Methods:

    def __init__(self, X_train, X_test, y_train, y_test, target_names, feature_names):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.target_names = target_names
        self.feature_names = feature_names

    def decision_tree(self, max_depth=None, min_samples_leaf=1, min_samples_split=2, criterion='gini', visualize_tree=False):

        DT = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split, criterion=criterion, random_state=0)

        DT.fit(self.X_train, self.y_train)
        y_pred_train = DT.predict(self.X_train)
        y_pred_val = DT.predict(self.X_test)

        if visualize_tree:

            export_graphviz(
            DT,
            out_file="Figures/DecisionTree_{}_{}.dot".format(criterion,max_depth),
            feature_names = self.feature_names,
            class_names = self.target_names,
            rounded=True,
            filled=True)
            cmd = 'dot -Tpng Figures/DecisionTree_{}_{}.dot -o Figures/DecisionTree_{}_{}.png'.format(criterion, max_depth, criterion, max_depth)
            #cmd = 'dot -Tpng Figures/DecisionTree_{}_{}.dot -o Figures/DecisionTree_{}.png'.format(max_depth, max_depth)
            os.system(cmd)

        return y_pred_train, y_pred_val, DT


    def gradient_boosting(self, n_estimators=100, criterion='friedman_mse', max_depth=3, min_samples_leaf=1, min_samples_split=2, learning_rate=0.01  ):
        GB = GradientBoostingClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, learning_rate=learning_rate)
        GB.fit(self.X_train, self.y_train)
        y_pred_train = GB.predict(self.X_train)
        y_pred_val = GB.predict(self.X_test)

        return y_pred_train, y_pred_val, GB

    def adaboost(self, base_estimator=None, n_estimators=50, learning_rate=1.0):
        AB = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, learning_rate=learning_rate, random_state=1)
        AB.fit(self.X_train, self.y_train)
        y_pred_train = AB.predict(self.X_train)
        y_pred_val = AB.predict(self.X_test)

        return y_pred_train, y_pred_val, AB

    def random_forest(self, n_estimators=100, criterion='gini', max_depth=None, min_samples_leaf=1, min_samples_split=2 ):

        RF = RandomForestClassifier(bootstrap=True, n_estimators=n_estimators, criterion=criterion,
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, random_state=0)

        RF.fit(self.X_train, self.y_train)



        y_pred_train = RF.predict(self.X_train)
        y_pred_val = RF.predict(self.X_test)

        return y_pred_train, y_pred_val, RF


    def neural_network(self, activation='relu', hidden_layer_sizes=(100,), max_iter=200, solver='adam', learning_rate=0.001):

        NN = MLPClassifier(activation=activation, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, solver=solver, learning_rate_init=learning_rate)

        NN.fit(self.X_train, self.y_train)

        y_pred_train = NN.predict(self.X_train)
        y_pred_val = NN.predict(self.X_test)

        return y_pred_train, y_pred_val, NN
