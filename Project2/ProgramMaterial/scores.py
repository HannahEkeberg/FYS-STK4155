import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def accuracy_function(y_data, y_pred):  #hvor bra modellen gjor det.
    if np.shape(y_data)[0] == np.shape(y_pred)[0]:
        I = np.mean(y_data==y_pred)     ## =1 if y==y, 0 else.... np.mean #dobbelsjekk at det funker
    else:
        raise ValuError
    return I

def accuracy_matrix(y_data, y_pred, title):
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_data, y_pred)
    print("confusion matrix")
    print(cm)

    precision_0 = cm[0,0]/(cm[0,0]+cm[1,0])
    precision_1 = cm[1,1]/(cm[1,1]+cm[0,1])
    recall_0 = cm[0,0]/(cm[0,0]+cm[0,1])
    recall_1 = cm[1,1]/(cm[1,1]+cm[1,0])
    print("precition 0:", precision_0)
    print("precition 1:", precision_1)
    print("recall 0:", recall_0)
    print("recall 1:", recall_1)

    cmap = plt.cm.Blues
    im = ax.imshow(cm, interpolation = 'nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks = np.arange(cm.shape[1]),
            yticks = np.arange(cm.shape[0]),
            title=title,
            xlabel='predicted target value',
            ylabel='true target value')
    name_file = title.strip()
    plt.savefig('Figures/Confusion_Matrix_{}.png'.format(name_file), dpi=300)
    plt.show()

    return cm


def MSE_func(y_data, y_pred): #for regression analysis
    return np.mean((y_data - y_pred)**2)

#def R2(y_data, y_pred):
#   return 1 - np.sum((y_data- y_pred)**2 / np.sum(y_data - np.mean(y_pred)**2))

def R2_func(y_data, y_pred):
    return 1 - ( np.sum((y_data-y_pred)**2) / np.sum((y_data-np.mean(y_pred))**2) )
