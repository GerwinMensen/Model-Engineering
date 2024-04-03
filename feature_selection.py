import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel




def feature_selection_TreeClassifier_SelectFromModel(X,y, threshold_importance):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X,y)
    model = SelectFromModel(clf,  threshold=threshold_importance)
    pd.DataFrame(model.estimator.feature_importances_).to_csv("feature_importance.csv")
    pd.DataFrame(model.estimator.feature_names_in_).to_csv("feature_importance_names.csv")
    model.set_output(transform="pandas")
    X_new = model.transform(X)
    return X_new

def feature_selection_TreeClassifier_KBest(X,y, k):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X,y)
    # model = SelectKBest(f_classif, k=k)
    model = SelectKBest(chi2, k=k)
    model.fit(X, y)
    model.set_output(transform="pandas")
    X_new = model.transform(X)
    return X_new











def feature_selection_chi2 (X, y, threshold):
    chi2_stats, p_values = chi2(X, y)
    for i in range( X.shape[1] -1, 0, -1):
        if p_values[i] < threshold:
            X = X.drop(X.columns[i], axis=1)
    return chi2_stats, p_values, X

def feature_selection_KBest (X, y, threshold):
    X_new = SelectKBest(f_classif, k=2).fit_transform(X, y)
    return X_new

def feature_selection_RFECV(X, y, target):
    estimator = RandomForestClassifier(random_state=0)
    selector = RFECV(estimator, step=1, scoring=target)
    selector.fit(X, y)
    selected_features = np.array(X.columns)[selector.get_support()]
    return selector