

from get_data import ReadData
from sklearn_methods import Methods
from analyse import Analyse
from tuning import *
from grid_searchCV import *

import matplotlib.colors as mcolors

import pandas as pd


from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

read = ReadData(data)
X_train, X_test, y_train, y_test = read.get_data(correlation_drop=True)
target_names, feature_names = read.get_data(correlation_drop=True, return_names=True)

method = Methods(X_train, X_test, y_train, y_test, target_names, feature_names)   # Sklearn Methods


# Each method return y_pred_train, y_pred_val and the actual method.
# These classifier_XX is assigned with a method.
classifier_rf = method.random_forest()[-1]
classifier_dt = method.decision_tree()[-1]
classifier_nn = method.neural_network(max_iter=1000)[-1]
classifier_ab = method.adaboost()[-1]



### From grid_searchCV function to get optimal parameters
### Does only print the optimal parameters from in grid_searchCV.py

#AdaBoost_params(classifier_ab, X_train, y_train)
#RandomForest_params(classifier_rf, X_train, y_train)
#DecisionTree_params(classifier_dt, X_train, y_train)
#NeuralNetwork_params(classifier_nn, X_train, y_train)




### METHODS RAN THROUGH tuning.py
### Did not include AdaBoost due to lack of time.
def tuning():
    #numb_trees(method, 'gradient_boosting', y_train, y_test)
    #numb_trees(method, 'random_forest', y_train, y_test)

    tree_depth(method, 'decision_tree', y_train, y_test)
    #tree_depth(method, 'gradient_boosting', y_train, y_test)
    #tree_depth(method, 'random_forest', y_train, y_test)

    #min_samples_leaf(method, 'decision_tree', y_train, y_test)
    #min_samples_leaf(method, 'random_forest', y_train, y_test)

    #min_samples_split(method, 'decision_tree', y_train, y_test)
    #min_samples_split(method, 'random_forest', y_train, y_test)

    max_iterations(method, 'neural_network', y_train, y_test, 400)

    #hidden_layers(method, y_train, y_test)

#tuning()    # calling the tuned parameters making plots.


## We did a test for both all predictors  (tuned_full_predictors()) and
## only predictors with low enough correlation (tuned_dropped_predictors()).
## Each method is assigned with the optimal parameters from the grid search.
def tuned_dropped_predictors():
    base_estimator=method.decision_tree(max_depth=1)
    RF_tuned = method.random_forest(n_estimators=150, max_depth=10, criterion='entropy', min_samples_leaf=2, min_samples_split=2)
    DT_tuned = method.decision_tree(criterion='gini', max_depth=3, min_samples_leaf=2, min_samples_split=2, visualize_tree=True)
    NN_tuned = method.neural_network(activation='relu', max_iter=100, learning_rate=0.1, hidden_layer_sizes=(20,))
    AB_tuned = method.adaboost(base_estimator=base_estimator[-1], n_estimators=200, learning_rate=0.1 )
    return RF_tuned, AB_tuned, DT_tuned, NN_tuned

def tuned_full_predictors():
    base_estimator=method.decision_tree(max_depth=2)
    RF_tuned = method.random_forest(n_estimators=30, max_depth=5, criterion='entropy', min_samples_leaf=2, min_samples_split=6)
    DT_tuned = method.decision_tree(criterion='gini', max_depth=10, min_samples_leaf=2, min_samples_split=2, visualize_tree=False)
    NN_tuned = method.neural_network(activation='tanh', max_iter=50, learning_rate=0.1, hidden_layer_sizes=(250,))
    AB_tuned = method.adaboost(base_estimator=base_estimator[-1], n_estimators=200, learning_rate=0.1 )

    return RF_tuned, AB_tuned, DT_tuned, NN_tuned


## Each method, with y_pred_train, y_pred_val, and method

#RF_tuned, AB_tuned, DT_tuned, NN_tuned = tuned_full_predictors()
RF_tuned, AB_tuned, DT_tuned, NN_tuned = tuned_dropped_predictors()


## Assigning analysis class to each classifier

anal_RF = Analyse(y_train, y_test, RF_tuned[0], RF_tuned[1])
anal_DT = Analyse(y_train, y_test, DT_tuned[0], DT_tuned[1])
anal_AB = Analyse(y_train, y_test, AB_tuned[0], AB_tuned[1])
anal_NN = Analyse(y_train, y_test, NN_tuned[0], NN_tuned[1])

## Getting the most important predictors for each method, as well as bar plot

DT = anal_DT.importance_predictors(DT_tuned[-1], feature_names, 'Feature importance - Decision tree', 'feature_importance_IP_DT')
RF = anal_RF.importance_predictors(RF_tuned[-1], feature_names, 'Feature importance - Random forest', 'feature_importance_IP_RT')
AB = anal_AB.importance_predictors(AB_tuned[-1], feature_names,'Feature importance - AnaBoost', 'feature_importance_IP_AB')

## Printing out accuracy and AUC

print("Accuracy scores:")
print("AdaBoost:", "train:", anal_AB.accuracy()[0], "val:", anal_AB.accuracy()[1])
print("Random forest:", "train:", anal_RF.accuracy()[0], "val:", anal_RF.accuracy()[1])
print("decision tree:", "train:", anal_DT.accuracy()[0], "val:", anal_DT.accuracy()[1])
print("neural network:", "train:", anal_NN.accuracy()[0], "val:", anal_NN.accuracy()[1])
#print("logistic regression:", "train:", anal_LR.accuracy()[0], "val:", anal_LR.accuracy()[1])

print("AUC scores:")
print("Adaboost:", "train:", anal_AB.roc()[-2], "val:", anal_AB.roc()[-1])
print("Random forest:", "train:", anal_RF.roc()[-2], "val:", anal_RF.roc()[-1])
print("decision tree:", "train:", anal_DT.roc()[-2], "val:", anal_DT.roc()[-1])
print("neural network:", "train:", anal_NN.roc()[-2], "val:", anal_NN.roc()[-1])
#print("logistic regression:", "train:", anal_LR.roc()[-2], "val:", anal_LR.roc()[-1])



## Plotting ROC curve. Training data was dropped due to confusing plot. Can be activated.

fpr_train_rf, tpr_train_rf, fpr_val_rf, tpr_val_rf = anal_RF.roc()[:4]
fpr_train_dt, tpr_train_dt, fpr_val_dt, tpr_val_dt = anal_DT.roc()[:4]
fpr_train_nn, tpr_train_nn, fpr_val_nn, tpr_val_nn = anal_NN.roc()[:4]
fpr_train_ab, tpr_train_ab, fpr_val_ab, tpr_val_ab = anal_AB.roc()[:4]


#plt.plot(fpr_train_dt, tpr_train_dt, linestyle='dashdot', linewidth=0.3, color='forestgreen',label='Decision tree - training data')
plt.plot(fpr_val_dt, tpr_val_dt, color='forestgreen', label='Decision tree - AUC = {:.4f}'.format(anal_DT.roc()[-1]))

#plt.plot(fpr_train_ab, tpr_train_ab, linestyle='dotted',linewidth=0.3, color='indianred', label='AdaBoost - training data')
plt.plot(fpr_val_ab, tpr_val_ab, color='indianred', linestyle='-.', label='AdaBoost - AUC = {:.4f}'.format(anal_AB.roc()[-1]))

#plt.plot(fpr_train_rf, tpr_train_rf, linestyle='dashed',color='purple',  label='Random forest - training data')
plt.plot(fpr_val_rf, tpr_val_rf, color='purple', linestyle=':', label='Random forest - AUC  = {:.4f}'.format(anal_RF.roc()[-1]))

#plt.plot(fpr_train_nn, tpr_train_nn, linestyle=':',linewidth=0.3, color='navy', label='Neural network - training data')
plt.plot(fpr_val_nn, tpr_val_nn, color='royalblue', label='Neural network - AUC = {:.4f}'.format(anal_NN.roc()[-1]))


plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC - with tuned parameters')
plt.legend()
plt.savefig('Figures/ROC_curve_dropped.png', dpi=300)
plt.show()



## Confusion matrices, and precision and recall for each class

cm_rf = anal_RF.confusion_matrix()[-1]
anal_RF.plot_confusion_matrix(cm_rf, 'Confusion matrix - random forest', 'CM_IP_RF')
print(cm_rf)

cm_dt = anal_DT.confusion_matrix()[-1]
anal_DT.plot_confusion_matrix(cm_dt, 'Confusion matrix - decision tree', 'CM_IP_DT')
print(cm_dt)

cm_ab = anal_AB.confusion_matrix()[-1]
anal_AB.plot_confusion_matrix(cm_ab, 'Confusion matrix - AdaBoost', 'CM_IP_AB')
print(cm_ab)

cm_nn = anal_NN.confusion_matrix()[-1]
anal_NN.plot_confusion_matrix(cm_nn, 'Confusion matrix - neural network', 'CM_IP_NN')
print(cm_nn)
