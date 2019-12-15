from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import os

from sklearn.tree import export_graphviz
from IPython.display import Image
from pydot import graph_from_dot_data


from analyse import Analyse

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
        min_samples_split=min_samples_split, criterion=criterion)

        DT.fit(self.X_train, self.y_train)
        y_pred_train = DT.predict(self.X_train)
        y_pred_val = DT.predict(self.X_test)

        #accuracy_train, accuracy_val = Analyse(self.y_train, self.y_test, y_pred_train, y_pred_val).accuracy()


        #print(accuracy_train, accuracy_val)

        if visualize_tree:

            export_graphviz(
            DT,
            out_file="Figures/DecisionTree_{}.dot".format(criterion),
            feature_names = self.feature_names,
            class_names = self.target_names,
            rounded=True,
            filled=True)
            cmd = 'dot -Tpng Figures/DecisionTree_{}.dot -o Figures/DecisionTree_{}.png'.format(criterion, criterion)
            os.system(cmd)

        return y_pred_train, y_pred_val


    def random_forest(self, n_estimators=100, criterion='gini', max_depth=None, min_samples_leaf=1, min_samples_split=2 ):

        RF = RandomForestClassifier(bootstrap=True, n_estimators=n_estimators, criterion=criterion,
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)

        RF.fit(self.X_train, self.y_train)

        y_pred_train = RF.predict(self.X_train)
        y_pred_val = RF.predict(self.X_test)

        return y_pred_train, y_pred_val


    def neural_network(self, activation='relu', hidden_layer_sizes=(100,), max_iter=200, solver='adam'):
        NN = MLPClassifier(activation=activation, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, solver=solver)
        NN.fit(self.X_train, self.y_train)

        y_pred_train = NN.predict(self.X_train)
        y_pred_val = NN.predict(self.X_test)

        return y_pred_train, y_pred_val 


    def logistic_regression(self, max_iter=100, solver='lbfgs'):
        LR = LogisticRegression(max_iter=max_iter, solver=solver)
        LR.fit(self.X_train, self.y_train)

        y_pred_train = LR.predict(self.X_train)
        y_pred_val = LR.predict(self.X_test)

        return y_pred_train, y_pred_val
