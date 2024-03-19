import numpy as np
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

def feature_selection_TreeClassifier(X,y, threshold_importance, target):
    clf = ExtraTreesClassifier()
    clf.fit(X, y)
    # print(clf_random.feature_importances_)
    model = SelectFromModel(clf, prefit=True, threshold=threshold_importance)
    model.set_output(transform="pandas")
    X_new = model.transform(X)
    return X_new