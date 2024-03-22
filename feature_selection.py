import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV

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

def feature_selection_TreeClassifier(X,y, threshold_importance):
    clf = ExtraTreesClassifier()
    clf.fit(X, y)
    # print(clf_random.feature_importances_)
    model = SelectFromModel(clf, prefit=True, threshold=threshold_importance)
    pd.DataFrame(model.estimator.feature_importances_).to_csv("feature_importance.csv")
    pd.DataFrame(model.estimator.feature_names_in_).to_csv("feature_importance_names.csv")
    # np.savetxt("feature_importance.csv", model.estimator.feature_importances_, delimiter=";")
    # np.savetxt("feature_importance_names.csv", model.estimator.feature_names_in_, delimiter=";")
    model.set_output(transform="pandas")
    X_new = model.transform(X)
    return X_new