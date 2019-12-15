

from get_data import ReadData
from sklearn_methods import Methods
from analyse import Analyse
from tuning import *


from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

read = ReadData(data)
X_train, X_test, y_train, y_test = read.get_data()
target_names, feature_names = read.get_data(return_names=True)


method = Methods(X_train, X_test, y_train, y_test, target_names, feature_names)



"""
Finding tuning parameters - results
Running gini:
Random forest: n_estimators=20, max_depth=11, min_samples_leaf=1, min_samples_split=2
Decision tree: max_depth=5, min_samples_leaf=1, min_samples_split=2

Running relu and with solver=adam:
neural network: max_iterations=300, hidden layers =200

running solver=lbfgs
Logistic regression: max_iter = 15
"""

"""
###METHODS RAN THROUGH tuning.py

#numb_trees(method, y_train, y_test)
#tree_depth(method, 'decision_tree', y_train, y_test)
#min_samples_leaf(DT_class, 'random_forest', y_train, y_test)
#min_samples_split(method, 'random_forest', y_train, y_test)

#max_iterations(method, 'neural_network', y_train, y_test, 400)
#max_iterations(method, 'logistic_regression', y_train, y_test, 40)

#hidden_layers(method, y_train, y_test)

"""

### Working with tuned parameters
RF_tuned = method.random_forest(n_estimators=20, max_depth=11)
DT_tuned = method.decision_tree(max_depth=5)
NN_tuned = method.neural_network(hidden_layer_sizes=(200,), max_iter=2000)
LR_tuned = method.logistic_regression(max_iter=15)

anal_RF = Analyse(y_train, y_test, RF_tuned[0], RF_tuned[1])
anal_DT = Analyse(y_train, y_test, DT_tuned[0], DT_tuned[1])
anal_NN = Analyse(y_train, y_test, NN_tuned[0], NN_tuned[1])
anal_LR = Analyse(y_train, y_test, LR_tuned[0], LR_tuned[1])


print("Accuracy scores:")
print("Random forest:", "train:", anal_RF.accuracy()[0], "val:", anal_RF.accuracy()[1])
print("decision tree:", "train:", anal_RF.accuracy()[0], "val:", anal_RF.accuracy()[1])
print("neural network:", "train:", anal_NN.accuracy()[0], "val:", anal_NN.accuracy()[1])
print("logistic regression:", "train:", anal_LR.accuracy()[0], "val:", anal_LR.accuracy()[1])

print("AUC scores:")
print("Random forest:", "train:", anal_RF.roc()[-2], "val:", anal_RF.roc()[-1])
print("decision tree:", "train:", anal_DT.roc()[-2], "val:", anal_DT.roc()[-1])
print("neural network:", "train:", anal_NN.roc()[-2], "val:", anal_NN.roc()[-1])
print("logistic regression:", "train:", anal_LR.roc()[-2], "val:", anal_LR.roc()[-1])

fpr_train_rf, tpr_train_rf, fpr_val_rf, tpr_val_rf = anal_RF.roc()[:4]
fpr_train_dt, tpr_train_dt, fpr_val_dt, tpr_val_dt = anal_DT.roc()[:4]
fpr_train_nn, tpr_train_nn, fpr_val_nn, tpr_val_nn = anal_NN.roc()[:4]
fpr_train_lr, tpr_train_lr, fpr_val_lr, tpr_val_lr = anal_LR.roc()[:4]


#plt.plot(fpr_train_rf, tpr_train_rf, '--', label='Random forest - training data')
plt.plot(fpr_val_rf, tpr_val_rf, label='Random forest - validation data')

#plt.plot(fpr_train_dt, tpr_train_dt, '--',label='Decision tree - training data')
plt.plot(fpr_val_dt, tpr_val_dt, label='Decision tree - validation data')

#plt.plot(fpr_train_nn, tpr_train_nn,'--', label='Neural network - training data')
plt.plot(fpr_val_nn, tpr_val_nn, label='Neural network - validation data')

#plt.plot(fpr_train_lr, tpr_train_lr, '--',label='Logistic regression - training data')
plt.plot(fpr_val_lr, tpr_val_lr, label='Logistic regression - validation data')


plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC - with tuned parameters')
plt.legend()
plt.savefig('Figures/ROC_curve.png', dpi=300)
plt.show()




cm_rf = anal_RF.confusion_matrix()[-1]
anal_RF.plot_confusion_matrix(cm_rf, 'Confusion matrix - random forest', 'CM_RF')
print(cm_rf)

cm_dt = anal_DT.confusion_matrix()[-1]
anal_DT.plot_confusion_matrix(cm_dt, 'Confusion matrix - decision tree', 'CM_DT')
print(cm_dt)

cm_nn = anal_NN.confusion_matrix()[-1]
anal_NN.plot_confusion_matrix(cm_nn, 'Confusion matrix - neural network', 'CM_NN')
print(cm_nn)

cm_lr = anal_LR.confusion_matrix()[-1]
anal_LR.plot_confusion_matrix(cm_lr, 'Confusion matrix - logistic regression', 'CM_LR')
print(cm_lr)
