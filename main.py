
import data_load as dl
import prepare_data as prepdata
import feature_importance as fi
import parameter_tuning as pt
import feature_selection as fs

# import packages
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from scipy.stats import uniform

from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer



# Variablen definieren
filename_credit_transactions = r'PSP_Jan_Feb_2019.xlsx'
target = "roc_auc"
# target = "f1"

# entweder die folgenden beiden Befehle
# Daten einlesen
# unprepared_df = dl.data_load(filename=filename_credit_transactions)
# Daten aufbereiten
# prepared_df = prepdata.prepare_data(unprepared_df)

# oder diese beiden Befehle
prepared_df = dl.data_load(filename='Datengrundlage.csv')
prepared_df = prepared_df.drop(columns='Unnamed: 0', axis=1)


# ID löschen
prepared_df = prepared_df.drop(columns='ID', axis=1)

# Unterteilen der vorbereiten Daten nach abh. und unabhängiger Variablen
# die Erfolgswahrscheinlichkeit soll prognostiziert werden --> Spalte success
X = prepared_df.drop(columns='success', axis=1)

# nur die numerischen Spalten behalten
X = X.select_dtypes(include=np.number)
X.to_csv('X_numerisch.csv', sep=';')
y = prepared_df.success



# chi2_stats, p_values, X_important_features = fs.feature_selection_chi2(X,y,0.001)
# selector = fs.feature_selection_RFECV(X, y, "f1")


# X_relevant = fs.feature_selection_TreeClassifier(X, y, "mean")
# X_relevant.to_csv('X_relevant.csv', sep=';')

X_relevant = X

# Unterteilen in Test- und Trainingsmenge
X_train, X_test, y_train, y_test = train_test_split(X_relevant, y, test_size=0.2, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X_relevant, y, test_size=0.3, random_state=42)

# X_train.to_csv('X_train.csv', sep=';')
# X_test.to_csv('X_test.csv', sep=';')


# Scaling the features using pipeline
pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])
scaled_X_train = pipeline.fit_transform(X_train)
scaled_X_test = pipeline.fit_transform(X_test)



"""
best_logistic = pt.logistic_regression_tuning(X_train,  y_train,  target)
print('Results logistic regression')
logistic_regression_result = pt.evaluate_model(best_logistic, X_train, X_test, y_train, y_test)
fi.show_feature_importance(best_logistic, X_test)
"""



best_decision_tree = pt.decision_tree_tuning(X_train,  y_train,  target)
print('Results decision tree')
decision_tree_result = pt.evaluate_model(best_decision_tree, X_train, X_test, y_train, y_test)
fi.show_feature_importance_tree(best_decision_tree, X_test)


"""
best_random_forest = pt.random_forest_tuning(X_train, y_train, target)
print('Results random forest')
random_forest_result = pt.evaluate_model(best_random_forest, X_train, X_test, y_train, y_test)
fi.show_feature_importance_tree(best_random_forest, X_test)
"""

"""
best_extra_tree = pt.extra_trees_tuning(X_train, y_train, target)
print('Results extra tree')
random_extra_tree = pt.evaluate_model(best_extra_tree, X_train, X_test, y_train, y_test)
fi.show_feature_importance_tree(best_extra_tree, X_test)





best_xg = pt.xgboost_tuning(X_train, y_train, target)
print('Results XGBoost')
xg_result = pt.evaluate_model(best_xg, X_train, X_test, y_train, y_test)
fi.show_feature_importance_tree(best_xg, X_test)
"""











best_svm = pt.svm_tuning(scaled_X_train, y_train, target)
print('Results SVM')
svm_result = pt.evaluate_model(best_svm, scaled_X_train, scaled_X_test, y_train, y_test)
fi.show_feature_importance(best_svm, X_test )











