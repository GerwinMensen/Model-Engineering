import pandas as pd
import numpy as np
from scipy.stats import uniform

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score,  classification_report, precision_score, f1_score, mean_squared_error, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from xgboost import XGBClassifier

from matplotlib import pyplot as plt



def evaluate_model(classifier, X_train, X_test, y_train, y_test):
    predictions_train = classifier.predict(X_train).round()
    predictions_test = classifier.predict(X_test).round()
    print("Train Accuracy :", accuracy_score(y_train, predictions_train))
    print("Train Recall :", recall_score(y_train, predictions_train))
    print("Train Precision :", precision_score(y_train, predictions_train))
    print("Train F1-Score :", f1_score(y_train, predictions_train))
    print("Train Roc_auc-Score :", roc_auc_score(y_train, predictions_train))
    print("Train Confusion Matrix:")
    print(confusion_matrix(y_train, predictions_train))
    print("-"*50)
    print("Test Accuracy :", accuracy_score(y_test, predictions_test))
    print("Test Recall :", recall_score(y_test, predictions_test))
    print("Test Precision :", precision_score(y_test, predictions_test))
    print("Test F1-Score :", f1_score(y_test, predictions_test))
    print("Test Roc_auc-Score :", roc_auc_score(y_test, predictions_test))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, predictions_test))
    return  {'Train accuracy': accuracy_score(y_train, predictions_train),
             'Train recall': recall_score(y_train, predictions_train, zero_division=np.nan),
             # 'Train precision': precision_score(y_train, predictions_train, zero_division=np.nan),
             'Train f1-score': f1_score(y_train, predictions_train, zero_division=np.nan),
             "Train Roc_auc-Score": roc_auc_score(y_train, predictions_train),
             'Train confusion matrix': confusion_matrix(y_train, predictions_train),
             'Test accuracy': accuracy_score(y_test, predictions_test),
             'Test recall': recall_score(y_test, predictions_test, zero_division=np.nan),
             'Test precision': precision_score(y_test, predictions_test, zero_division=np.nan),
             'Test f1-score': f1_score(y_test, predictions_test, zero_division=np.nan),
             "Test Roc_auc-Score": roc_auc_score(y_test, predictions_test),
             'Test confusion matrix': confusion_matrix(y_test, predictions_test)
             }




def logistic_regression_tuning (X_train, y_train,  target):
    logistic = LogisticRegression()
    max_iter = [10000]
    penalty = ['none', 'l2', 'l1']
    solver = ['lbfgs', 'liblinear']
    C = np.arange(0, 1, 0.01)
    random_grid = {
        'penalty': penalty,
        'solver': solver,
        'max_iter': max_iter,
        'C': C,
    }
    clf = RandomizedSearchCV(estimator=logistic, param_distributions=random_grid, cv=5, verbose=1, random_state=0, scoring = target)
    search = clf.fit(X_train, y_train)
    print(search.best_params_)
    # Ergebnis logistische Regression
    logistic_best = search.best_estimator_
    return logistic_best

def decision_tree_tuning (X_train, y_train,  target):
    # Create the parameter grid based on the results of random search
    dt = DecisionTreeClassifier(random_state=42)
    path = dt.cost_complexity_pruning_path
    params = {
        'criterion': ["gini", "entropy"],
        'max_depth': np.arange(5, 11, 1),
        'min_samples_split': np.arange(2, 4, 1),
        'max_leaf_nodes': np.arange(30, 81, 10),
        'min_samples_leaf': np.arange(10, 101, 10)
    }
    grid_search = RandomizedSearchCV(estimator=dt,param_distributions=params, cv=5, n_jobs=-1, verbose=1, scoring = target)
    # grid_search = GridSearchCV(estimator=dt,param_grid=params, cv=5, n_jobs=-1, verbose=1, scoring = target)
    grid_search.fit(X_train, y_train)
    score_df = pd.DataFrame(grid_search.cv_results_)
    score_df.nlargest(5, "mean_test_score")
    grid_search.best_estimator_
    dt_best = grid_search.best_estimator_
    print(export_text(dt_best, feature_names=[f"{col}" for col in X_train.columns]))
    print('Best Parameters:', grid_search.best_params_, ' \n')
    plot_tree(dt_best)
    plt.show()
    return dt_best

def random_forest_tuning (X_train, y_train,  target):

    random_grid = {
                    'n_estimators': np.arange(80, 131, 10),
                   'max_features': ['sqrt', 'log2'],

                    # 'max_leaf_nodes': [5, 10, 20, 25, 30, 40, 50],
                    'criterion': ["gini", "entropy"],
                    'max_depth': np.arange(2, 16, 3),
                    'min_samples_split': np.arange(2, 16, 3),
                    'min_samples_leaf': np.arange(20, 101, 20),
                   # 'max_depth': [5, 10, 20, 30, 40, 50, 60, 70],
                   # 'min_samples_split': [5, 10, 20, 25, 30, 40, 50],
                   # 'min_samples_leaf': [1, 100, 200, 300, 400, 500]
                  }

    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=rf, cv=5, param_distributions=random_grid, n_jobs=-1, verbose=1, scoring=target)
                                  #  n_iter=100,  random_state=42)
    rf_random.fit(X_train, y_train)
    # print the best parameters
    print('Best Parameters:', rf_random.best_params_, ' \n')
    rf_best = rf_random.best_estimator_
    # result_rf = evaluate_model(rf_best, X_train, X_test, y_train, y_test)
    return rf_best




def svm_tuning (scaled_X_train,  y_train,  target):
    svm = SVC()
    param_grid = {'C':[0.01,0.05,0.1,1,10, 100, 1000],'kernel':['linear','rbf'], 'gamma':['scale','auto'] }
    """
    param_grid = {
                  # 'C': [0.1, 1, 10, 100, 1000],
                  'C': [0.1, 1],
                  # 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'gamma': [1],
                  # 'kernel': ['rbf', 'linear']
                  'kernel': ['linear']}
    """
    grid = RandomizedSearchCV(svm, param_distributions=param_grid, n_jobs=-1, cv=5, verbose=1, scoring = target)
    # grid = GridSearchCV(svm, param_grid, cv=5, scoring = target)
    # Fitting the model
    grid.fit(scaled_X_train, y_train)
    svm_best = grid.best_estimator_
    # Classification report for the tuned model
    # print(classification_report(y_test, grid_svc))
    print(grid.best_params_)
    print(grid.best_estimator_.get_params())
    return svm_best


def extra_trees_tuning(X_train,  y_train,  target):

    random_grid = {'n_estimators': [100, 200, 300, 400, 500],
                   'max_depth': [5, 10, 20, 30, 40, 50, 60, 70],
                   'min_samples_split': [5, 10, 20, 25, 30, 40, 50],
                   'max_features': ['sqrt', 'log2'],
                   'max_leaf_nodes': [5, 10, 20, 25, 30, 40, 50],
                   'min_samples_leaf': [1, 100, 200, 300, 400, 500]
                  }

    extra_tree_clf = ExtraTreesClassifier()
    extra_tree_random = RandomizedSearchCV(estimator=extra_tree_clf, n_jobs=-1, cv=5, verbose=1, param_distributions=random_grid, scoring=target)
    extra_tree_random.fit(X_train,  y_train)
    print('Best Parameters:', extra_tree_random.best_params_, ' \n')
    extra_tree_best = extra_tree_random.best_estimator_
    return extra_tree_best

def xgboost_tuning (X_train,  y_train,  target):

    # declare parameters
    params = {
        'n_estimators': [100, 200, 300, 400, 500],
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }
    folds = 5
    param_comb = 5
    xgb = XGBClassifier(learning_rate=0.02, objective='binary:logistic', # , n_estimators=100
                        silent=True, nthread=1)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
    xg_random = RandomizedSearchCV(xgb, param_distributions=params, cv=5, n_iter=param_comb, scoring=target, n_jobs=-1,
                                        verbose=1, random_state=1001)
    xg_random.fit(X_train, y_train)
    print('Best Parameters:', xg_random.best_params_, ' \n')
    xg_best = xg_random.best_estimator_
    return xg_best