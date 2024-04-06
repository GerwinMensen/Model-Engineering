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
    model = SelectKBest(chi2, k=k)
    model.fit(X, y)
    model.set_output(transform="pandas")
    X_new = model.transform(X)
    result = pd.DataFrame()
    result['feature_name'] = pd.DataFrame(model.feature_names_in_)
    result['scores'] = pd.DataFrame(model.scores_)
    print(result)
    return X_new, result


