from sklearn.model_selection import GridSearchCV

def grid_search(grid_param, classifier, X_train, y_train):

        gd_sr = GridSearchCV(estimator = classifier, param_grid=grid_param,
        cv = 5, n_jobs=-1)

        gd_sr.fit(X_train, y_train)
        best_parameters = gd_sr.best_params_
        print(best_parameters)

def GradientBoosting_params(classifier, X_train, y_train):
    grid_param = {'n_estimators': [10,30, 70, 150],
                'max_depth': [2,3,5,10,15,20,30],
                'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
                'min_samples_leaf': [0.1,0.25,  0.5, 1,2,3],
                'min_samples_split': [0.1,0.5,2,4, 6]}

    grid_search(grid_param, classifier, X_train, y_train)

def RandomForest_params(classifier, X_train, y_train):
    grid_param = {'n_estimators': [10,30, 70, 150],
                'max_depth': [2,3,5,10,15,20,30],
                'min_samples_leaf': [0.1,0.25,0.5, 1,2,3],
                'min_samples_split': [0.1,0.5,2,4, 6],
                'criterion': ['gini', 'entropy']}

    grid_search(grid_param, classifier, X_train, y_train)

def DecisionTree_params(classifier, X_train, y_train):
    grid_param = {'max_depth': [2,3,5,10,15,20,30],
    'min_samples_leaf': [0.1,0.25,  0.5, 1,2,3],
    'min_samples_split': [0.1,0.5,2,4, 6],
    'criterion': ['gini', 'entropy']}

    grid_search(grid_param, classifier, X_train, y_train)


def NeuralNetwork_params(classifier, X_train, y_train):
    grid_param = {'max_iter': [10,20,50,100, 200, 400],
                'hidden_layer_sizes': [1,4,10,20,50,100, 200, 250],
                'activation': ['logistic', 'tanh', 'relu'],
                'learning_rate_init': [0.001, 0.01, 0.1, 1.0]}

    grid_search(grid_param, classifier, X_train, y_train)

