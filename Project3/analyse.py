from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

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
