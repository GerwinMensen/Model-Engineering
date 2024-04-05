import pandas as pd
import numpy as np

# Import relevanter sklearn packages
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.dummy import DummyClassifier

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

def baseline_model (X_train, y_train, X_test, y_test):
    model_baseline = DummyClassifier(strategy='most_frequent')
    model_baseline.fit(X_train, y_train)
    return model_baseline


def logistic_regression_tuning (X_train, y_train,  target):
    logistic = LogisticRegression()
    max_iter = [10000]
    penalty = [None, 'l2', 'l1']
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

    dt = DecisionTreeClassifier(random_state=42)
    path = dt.cost_complexity_pruning_path
    random_grid = {
        'criterion': ["gini", "entropy"],
        'max_depth': np.arange(5, 15, 1),
        'min_samples_split': np.arange(2, 4, 1),
        'max_leaf_nodes': np.arange(30, 101, 10),
        'min_samples_leaf': np.arange(10, 101, 10),
        'max_features': ['sqrt', 'log2', None]
    }
    dt_random = RandomizedSearchCV(estimator=dt,param_distributions=random_grid, cv=5, n_jobs=-1, verbose=1, scoring = target)
    dt_random.fit(X_train, y_train)
    score_df = pd.DataFrame(dt_random.cv_results_)
    score_df.nlargest(5, "mean_test_score")
    dt_random.best_estimator_
    dt_best = dt_random.best_estimator_
    print(export_text(dt_best, feature_names=[f"{col}" for col in X_train.columns]))
    print('Best Parameters:', dt_random.best_params_, ' \n')
    plot_tree(dt_best)
    plt.show()
    return dt_best

def random_forest_tuning (X_train, y_train,  target):

    random_grid = {
                    'n_estimators': np.arange(80, 131, 10),
                    'max_features': np.arange(8, 16, 1),
                    'criterion': ["gini", "entropy"],
                    'max_depth': np.arange(8, 15, 1),
                    'min_samples_split': np.arange(2, 5, 1),
                    'max_leaf_nodes': np.arange(50, 101, 10),
                    'min_samples_leaf': np.arange(20, 60, 10)
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
    random_grid = {'C':[1,10],
                  'kernel':['linear'],
                  'gamma':['auto'] }
    """
    random_grid = {'C':[0.01,0.05,0.1,1,10, 100, 1000],
                  'kernel':['linear','rbf'],
                  'gamma':['scale','auto'] }
    """
    svm_random = RandomizedSearchCV(svm, param_distributions=random_grid, n_jobs=-1, cv=5, verbose=1, scoring = target)
    # Fitting the model
    svm_random.fit(scaled_X_train, y_train)
    svm_best = svm_random.best_estimator_
    print(svm_random.best_params_)
    print(svm_random.best_estimator_.get_params())
    return svm_best


def extra_trees_tuning(X_train,  y_train,  target):
    random_grid  = {
        'n_estimators': [100, 200, 300, 400, 500],
        'criterion': ["gini", "entropy"],
        'max_depth': np.arange(5, 11, 1),
        'min_samples_split': np.arange(2, 4, 1),
        'max_leaf_nodes': np.arange(30, 81, 10),
        'min_samples_leaf': np.arange(10, 101, 10),
        'max_features': ['sqrt', 'log2', None]
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
        'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth':  np.arange(5, 11, 1)
    }
    folds = 5
    param_comb = 5
    xgb = XGBClassifier(learning_rate=0.02, objective='binary:logistic', metric = 'auc', # , n_estimators=100
                        silent=True, nthread=1)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
    xg_random = RandomizedSearchCV(xgb, param_distributions=params, cv=5, n_iter=param_comb, scoring=target, n_jobs=-1,
                                        verbose=1,  random_state=1001)
    xg_random.fit(X_train, y_train)
    print('Best Parameters:', xg_random.best_params_, ' \n')
    xg_best = xg_random.best_estimator_
    return xg_best