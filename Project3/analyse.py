from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
The program contains methods for measuring how well the different
algorithms perform, based upon training and test data.

* Accuracy returns the accuracy score.
* roc returns false positve and true positive rate, along with area under curve
* confusion matrix  returns the confusion matrix, and plot_confusion_matrix
gives a plot of the confusion matrix, along with precision and recall of class 0 and 1.
* importance_predictors prints the most important predictor as well as a bar plot of
the importance of the predictors.

"""


class Analyse:

    def __init__(self, y_train, y_test, y_pred_train, y_pred_val):
        self.y_train      = y_train
        self.y_test       = y_test
        self.y_pred_train = y_pred_train
        self.y_pred_val   = y_pred_val


    def accuracy(self):

        accuracy_train = accuracy_score(self.y_train, self.y_pred_train)
        accuracy_val   = accuracy_score(self.y_test, self.y_pred_val)

        return accuracy_train, accuracy_val

    def roc(self):

        fpr_train, tpr_train, threshold_train = roc_curve(self.y_train, self.y_pred_train)
        fpr_val, tpr_val, threshold_val = roc_curve(self.y_test, self.y_pred_val)

        auc_train = auc(fpr_train, tpr_train)
        auc_val   = auc(fpr_val, tpr_val)         #AREA UNDER CURVE

        return fpr_train, tpr_train, fpr_val, tpr_val, auc_train, auc_val


    def confusion_matrix(self):
        cm_train = confusion_matrix(self.y_train, self.y_pred_train)
        cm_val   = confusion_matrix(self.y_test, self.y_pred_val)

        return cm_train, cm_val


    def plot_confusion_matrix(self, cm, title, name_fig ):
        precision_0 = cm[0,0]/(cm[0,0]+cm[1,0])
        precision_1 = cm[1,1]/(cm[1,1]+cm[0,1])
        recall_0 = cm[0,0]/(cm[0,0]+cm[0,1])
        recall_1 = cm[1,1]/(cm[1,1]+cm[1,0])
        print("precition 0:", precision_0)
        print("precition 1:", precision_1)
        print("recall 0:", recall_0)
        print("recall 1:", recall_1)
        fig, ax = plt.subplots()
        cmap = plt.cm.Blues
        im = ax.imshow(cm, interpolation = 'nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks = np.arange(cm.shape[1]),
                yticks = np.arange(cm.shape[0]),
                title=title,
                xlabel='predicted target value',
                ylabel='true target value')

        plt.savefig('Figures/' + name_fig +'.png', dpi=300)
        plt.show()

    def importance_predictors(self, method, feature_names, title, name_fig):
        importance = method.feature_importances_
        importance = pd.DataFrame(importance, index = feature_names, columns=["importance"] )
        n = importance.shape[0]
        x = np.linspace(0,n,n)

        y = importance.ix[:, 0]
        print("Most important feature:", y.idxmax(), y.max())

        fig, ax = plt.subplots()
        ax.set_label('Scores')
        ax.set_xticks(x)

        names = np.linspace(1,len(feature_names), len(feature_names), dtype=int)
        ax.set_xticklabels(names,  horizontalalignment='right', fontsize = 8)
        plt.bar(x, y,align="center")
        plt.title(title)
        plt.savefig('Figures/'+name_fig+'.png', dpi=300)
        plt.show()





    #def plot_importance_predictors():
