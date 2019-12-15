
from analyse import Analyse
import matplotlib.pyplot as plt
import numpy as np

def make_plot(x_axis, train_accuracy, val_accuracy, train_auc, val_auc, xlabel, title, name_fig):
    plt.plot(x_axis, train_accuracy, label='train - accuracy')
    plt.plot(x_axis, val_accuracy, label='val - accuracy')
    plt.plot(x_axis, train_auc, label='train - AUC')
    plt.plot(x_axis, val_auc, label='val - AUC')
    plt.xlabel(xlabel)
    plt.title(title)
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('Figures/' + name_fig + '.png', dpi=300)
    plt.show()

def numb_trees(class_, y_train, y_test):
    trees = [1,3,5, 10, 15, 20, 50, 100, 150]
    train_accuracy = []; val_accuracy = []
    train_auc = []; val_auc = []
    for i in trees:
        y_pred_train, y_pred_val = class_.random_forest(n_estimators=i)

        anal = Analyse(y_train, y_test, y_pred_train, y_pred_val)
        accuracy_train, accuracy_val  = anal.accuracy()
        auc_train, auc_val = anal.roc()[4:]
        train_accuracy.append(accuracy_train)
        val_accuracy.append(accuracy_val)
        train_auc.append(auc_train)
        val_auc.append(auc_val)

    xlabel='Number of trees'; title='Random forest - number of trees'; name_fig='RF_numbTrees'
    make_plot(trees, train_accuracy, val_accuracy, train_auc, val_auc, xlabel, title, name_fig)

def tree_depth(class_, method, y_train, y_test):
    depth = [2,3,5,7,10,13,15,18, 20,30]
    train_accuracy = []; val_accuracy = []
    train_auc = []; val_auc = []

    for i in depth:
        if method == 'random_forest':
            y_pred_train, y_pred_val = class_.random_forest(max_depth=i)
            xlabel = 'max depth'; title='Random forest - tree depth'; name_fig='RF_TreeDepth'
        if method == 'decision_tree':
            y_pred_train, y_pred_val = class_.decision_tree(max_depth=i)
            xlabel = 'max depth'; title='Decision tree - tree depth'; name_fig='DT_TreeDepth'

        anal = Analyse(y_train, y_test, y_pred_train, y_pred_val)
        accuracy_train, accuracy_val  = anal.accuracy()
        auc_train, auc_val = anal.roc()[4:]
        train_accuracy.append(accuracy_train)
        val_accuracy.append(accuracy_val)
        train_auc.append(auc_train)
        val_auc.append(auc_val)

    make_plot(depth, train_accuracy, val_accuracy, train_auc, val_auc, xlabel, title, name_fig)


def min_samples_leaf(class_, method, y_train, y_test):
    samples_leaf = [0.1,0.25,  0.5, 1,2,3]
    #samples_leaf = np.linspace(0.1,0.05*n,n)
    train_accuracy = []; val_accuracy = []
    train_auc = []; val_auc = []
    for i in samples_leaf:
        if method == 'decision_tree':
            y_pred_train, y_pred_val = class_.random_forest(min_samples_leaf=i)
            xlabel = 'min samples leaf'; title='Decision tree - min samples leaf'; name_fig='DT_min_samples_leaf'
        elif method == 'random_forest':
            y_pred_train, y_pred_val = class_.random_forest(min_samples_leaf=i)
            xlabel = 'min samples leaf'; title='Random forest - min samples leaf'; name_fig='RF_min_samples_leaf'

        anal = Analyse(y_train, y_test, y_pred_train, y_pred_val)
        accuracy_train, accuracy_val  = anal.accuracy()
        auc_train, auc_val = anal.roc()[4:]
        train_accuracy.append(accuracy_train)
        val_accuracy.append(accuracy_val)
        train_auc.append(auc_train)
        val_auc.append(auc_val)

    make_plot(samples_leaf, train_accuracy, val_accuracy, train_auc, val_auc, xlabel, title, name_fig)


def min_samples_split(class_, method, y_train, y_test):
    samples_split = [0.1,0.5,2,4, 6]
    #samples_leaf = np.linspace(0.1,0.05*n,n)
    train_accuracy = []; val_accuracy = []
    train_auc = []; val_auc = []
    for i in samples_split:
        if method == 'decision_tree':
            y_pred_train, y_pred_val = class_.random_forest(min_samples_split=i)
            xlabel = 'min samples split'; title='Decision tree - min samples spit'; name_fig='DT_min_samples_split'
        elif method == 'random_forest':
            y_pred_train, y_pred_val = class_.random_forest(min_samples_split=i)
            xlabel = 'min samples split'; title='Random forest - min samples split'; name_fig='RF_min_samples_split'

        anal = Analyse(y_train, y_test, y_pred_train, y_pred_val)
        accuracy_train, accuracy_val  = anal.accuracy()
        auc_train, auc_val = anal.roc()[4:]
        train_accuracy.append(accuracy_train)
        val_accuracy.append(accuracy_val)
        train_auc.append(auc_train)
        val_auc.append(auc_val)

    make_plot(samples_split, train_accuracy, val_accuracy, train_auc, val_auc, xlabel, title, name_fig)


def max_iterations(class_, method, y_train, y_test, max_it):

    iterations = np.linspace(1,max_it, dtype=int)
    train_accuracy = []; val_accuracy = []
    train_auc = []; val_auc = []
    for i in iterations:
        if method == 'neural_network':
            y_pred_train, y_pred_val = class_.neural_network(max_iter=i)
            xlabel = 'max iterations'; title='Neural network - max iterations'; name_fig='NN_max_iterations'
        elif method == 'logistic_regression':
            y_pred_train, y_pred_val = class_.logistic_regression(max_iter=i)
            xlabel = 'max iterations'; title='Logistic regression- max iterations'; name_fig='LR_max_iterations'

        anal = Analyse(y_train, y_test, y_pred_train, y_pred_val)
        accuracy_train, accuracy_val  = anal.accuracy()
        auc_train, auc_val = anal.roc()[4:]
        train_accuracy.append(accuracy_train)
        val_accuracy.append(accuracy_val)
        train_auc.append(auc_train)
        val_auc.append(auc_val)

    make_plot(iterations, train_accuracy, val_accuracy, train_auc, val_auc, xlabel, title, name_fig)


def hidden_layers(class_, y_train, y_test):
    layers = [1,4,10,20,50,100, 200, 250]
    train_accuracy = []; val_accuracy = []
    train_auc = []; val_auc = []

    for i in layers:
        y_pred_train, y_pred_val = class_.neural_network(hidden_layer_sizes=i)

        anal = Analyse(y_train, y_test, y_pred_train, y_pred_val)
        accuracy_train, accuracy_val  = anal.accuracy()
        auc_train, auc_val = anal.roc()[4:]
        train_accuracy.append(accuracy_train)
        val_accuracy.append(accuracy_val)
        train_auc.append(auc_train)
        val_auc.append(auc_val)
    xlabel='hidden layers'; title='Neural network - hidden layers'; name_fig='NN_hiddenlayers'

    make_plot(layers, train_accuracy, val_accuracy, train_auc, val_auc, xlabel, title, name_fig)
