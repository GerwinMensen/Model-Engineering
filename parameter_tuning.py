import pandas as pd
import numpy as np
from scipy.stats import uniform

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score,  classification_report, precision_score, f1_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from matplotlib import pyplot as plt

def evaluate_model(classifier, X_train, X_test, y_train, y_test):
    predictions_train = classifier.predict(X_train).round()
    predictions_test = classifier.predict(X_test).round()
    print("Train Accuracy :", accuracy_score(y_train, predictions_train))
    print("Train Recall :", recall_score(y_train, predictions_train))
    print("Train Precision :", precision_score(y_train, predictions_train))
    print("Train F1-Score :", f1_score(y_train, predictions_train))
    print("Train Confusion Matrix:")
    print(confusion_matrix(y_train, predictions_train))
    print("-"*50)
    print("Test Accuracy :", accuracy_score(y_test, predictions_test))
    print("Test Recall :", recall_score(y_test, predictions_test))
    print("Test Precision :", precision_score(y_test, predictions_test))
    print("Test F1-Score :", f1_score(y_test, predictions_test))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, predictions_test))
    return  {'Train accuracy': accuracy_score(y_train, predictions_train),
             'Train recall': recall_score(y_train, predictions_train, zero_division=np.nan),
             'Train precision': precision_score(y_train, predictions_train, zero_division=np.nan),
             'Train f1-score': f1_score(y_train, predictions_train, zero_division=np.nan),
             'Train confusion matrix': confusion_matrix(y_train, predictions_train),
             'Test accuracy': accuracy_score(y_test, predictions_test),
             'Test recall': recall_score(y_test, predictions_test, zero_division=np.nan),
             'Test precision': precision_score(y_test, predictions_test, zero_division=np.nan),
             'Test f1-score': f1_score(y_test, predictions_test, zero_division=np.nan),
             'Test confusion matrix': confusion_matrix(y_test, predictions_test)
             }

def show_feature_importance (classifier, X_test):
    # Bedeutung der Variablen bei der logistischen Regression anzeigen
    coefficients = classifier.coef_[0]
    feature_importance = pd.DataFrame({'Feature': X_test.columns, 'Importance': np.abs(coefficients)})
    feature_importance = feature_importance.sort_values('Importance', ascending=True)
    feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))
    plt.show()


def logistic_regression_tuning (X_train, y_train,  target):
    logistic = LogisticRegression(max_iter=10000)
    distributions = dict(C=uniform(loc=0, scale=4), penalty=['l2'])

    clf = RandomizedSearchCV(estimator=logistic, param_distributions=distributions, random_state=0, scoring = target)
    search = clf.fit(X_train, y_train)
    print(search.best_params_)
    # Ergebnis logistische Regression
    logistic_best = search.best_estimator_
    return logistic_best

def decision_tree_tuning (X_train, y_train,  target):
    # Create the parameter grid based on the results of random search
    dt = DecisionTreeClassifier(random_state=42)
    params = {
        'max_depth': [2, 3, 5, 10, 20],
        'min_samples_leaf': [5, 10, 20, 50, 100],
        'criterion': ["gini", "entropy"]
    }

    grid_search = GridSearchCV(estimator=dt,param_grid=params, cv=5, n_jobs=-1, verbose=1, scoring = target)
    grid_search.fit(X_train, y_train)
    score_df = pd.DataFrame(grid_search.cv_results_)
    score_df.nlargest(5, "mean_test_score")
    grid_search.best_estimator_
    dt_best = grid_search.best_estimator_
    # result_decision_tree = evaluate_model(dt_best, X_train, X_test, y_train, y_test)
    return dt_best

def svm_tuning (scaled_X_train,  y_train,  target):
    svm = SVC()
    # param_grid = {'C':[0.01,0.05,0.1,1,10, 100, 1000],'kernel':['linear','rbf'], 'gamma':['scale','auto'] }
    param_grid = {
                  # 'C': [0.1, 1, 10, 100, 1000],
                  'C': [0.1, 1],
                  # 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'gamma': [1],
                  # 'kernel': ['rbf', 'linear']
                  'kernel': ['linear']}
    grid = GridSearchCV(svm, param_grid, cv=5, scoring = target)
    # Fitting the model
    grid.fit(scaled_X_train, y_train)
    svm_best = grid.best_estimator_
    # Classification report for the tuned model
    # print(classification_report(y_test, grid_svc))
    print(grid.best_params_)
    print(grid.best_estimator_.get_params())
    return svm_best

def random_forest_tuning (X_train, y_train,  target):

    random_grid = {'n_estimators': [100, 200, 300, 400, 500],
                   'max_depth': [5, 10, 20, 30, 40, 50, 60, 70],
                   'min_samples_split': [5, 10, 20, 25, 30, 40, 50],
                   'max_features': ['sqrt', 'log2'],
                   'max_leaf_nodes': [5, 10, 20, 25, 30, 40, 50],
                   'min_samples_leaf': [1, 100, 200, 300, 400, 500]
                  }

    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=rf, cv=5, param_distributions=random_grid, scoring=target)
                                  #  n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)
    rf_random.fit(X_train, y_train)
    # print the best parameters
    print('Best Parameters:', rf_random.best_params_, ' \n')
    rf_best = rf_random.best_estimator_
    # result_rf = evaluate_model(rf_best, X_train, X_test, y_train, y_test)
    return rf_best

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
    xg_random = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring=target, n_jobs=4,
                                       cv=skf.split(X_train, y_train), verbose=3, random_state=1001)
    xg_random.fit(X_train, y_train)
    print('Best Parameters:', xg_random.best_params_, ' \n')
    xg_best = xg_random.best_estimator_
    return xg_best