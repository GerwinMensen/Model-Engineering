
import data_load as dl
import prepare_data as prepdata
import feature_importance as fi
import parameter_tuning as pt

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

"""
names = [
    "Decision Tree",
    "Nearest Neighbors",
    "Linear SVM",
#    "RBF SVM",
#    "Gaussian Process",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]





classifiers = [
    DecisionTreeClassifier(max_depth=5, random_state=42),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=42),
#    SVC(gamma=2, C=1, random_state=42),
#    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    AdaBoostClassifier(algorithm="SAMME", random_state=42),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]
"""






# Variablen definieren
filename_credit_transactions = r'PSP_Jan_Feb_2019.xlsx'
target = "f1"
# Daten einlesen
unprepared_df = dl.data_load(filename=filename_credit_transactions)
# Daten aufbereiten
prepared_df = prepdata.prepare_data(unprepared_df)

# prepared_df = dl.data_load(filename='Datengrundlage.csv')
# prepared_df = prepared_df.drop(columns='Unnamed: 0', axis=1)
prepared_df = prepared_df.drop(columns='ID', axis=1)

# Unterteilen der vorbereiten Daten nach abh. und unabhängiger Variablen
# die Erfolgswahrscheinlichkeit soll prognostiziert werden --> Spalte success
X = prepared_df.drop(columns='success', axis=1)

# nur die numerischen Spalten behalten
X = X.select_dtypes(include=np.number)
# X.to_csv('X.csv', sep=';')
y = prepared_df.success

# Unterteilen in Test- und Trainingsmenge
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X_train.to_csv('X_train.csv', sep=';')
# X_test.to_csv('X_test.csv', sep=';')


# Scaling the features using pipeline
pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])
scaled_X_train = pipeline.fit_transform(X_train)
scaled_X_test = pipeline.fit_transform(X_test)

best_logistic = pt.logistic_regression_tuning(X_train,  y_train,  "f1")
print('Results logistic regression')
logistic_regression_result = pt.evaluate_model(best_logistic, X_train, X_test, y_train, y_test)
pt.show_feature_importance(best_logistic, X_test)

best_decision_tree = pt.decision_tree_tuning(X_train,  y_train,  "f1")
print('Results decision tree')
decision_tree_result = pt.evaluate_model(best_decision_tree, X_train, X_test, y_train, y_test)

best_random_forest = pt.random_forest_tuning(X_train, y_train, "f1")
print('Results random forest')
random_forest_result = pt.evaluate_model(best_random_forest, X_train, X_test, y_train, y_test)

best_svm = pt.svm_tuning(scaled_X_train, y_train, "f1")
print('Results SVM')
svm_result = pt.evaluate_model(best_svm, scaled_X_train, scaled_X_test, y_train, y_test)
pt.show_feature_importance(best_svm, X_test )


best_xg = pt.xgboost_tuning(X_train, y_train, "f1")
print('Results XGBoost')
xg_result = pt.evaluate_model(best_xg, X_train, X_test, y_train, y_test)






# Logistische Regression durchführen
# logistic = LogisticRegression(random_state=0).fit(X_scaled, y)

"""
logistic = LogisticRegression(max_iter=10000)
logistic.fit(X_train, y_train)
distributions = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'])


clf = RandomizedSearchCV(estimator=logistic,param_distributions=distributions, random_state=0)
search = clf.fit(X_train, y_train)
print(search.best_params_)
print(clf.score(X_test,y_test))
# Ergebnis logistische Regression

print("Genauigkeit")
predicted_logistic_regression = logistic.predict(X_test)
print(np.sum(predicted_logistic_regression * 1 == y_test) / len(X_test))

# Bedeutung der Variablen bei der logistischen Regression anzeigen
coefficients = logistic.coef_[0]
feature_importance = pd.DataFrame({'Feature': X_test.columns, 'Importance': np.abs(coefficients)})
feature_importance = feature_importance.sort_values('Importance', ascending=True)
feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))
plt.show()




# print(max_importance)
score_df = pd.DataFrame(columns=['classifier', 'score'])
# über Klassifizierungsmethoden iterieren
"""







"""
i = 0
for name, clf in zip(names, classifiers):
    clf = make_pipeline(StandardScaler(), clf)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    score_df.loc[i] = [name, score]

    clf_best = RandomizedSearchCV(estimator=clf, param_distributions=distributions, random_state=0)
    search = clf_best.fit(X_train, y_train)
    print(search.best_params_)
    print(clf_best.score(X_test, y_test))



    if name == 'Linear SVM':
        pd.Series(abs(clf.named_steps.svc.coef_[0]), index=X_train.columns).nlargest(10).plot(kind='barh')
        plt.show()
        # fi.f_importances(clf.named_steps.svc.coef_, X_train.columns)
    if name == 'Decision Tree':
        text_representation = tree.export_text(clf.named_steps.decisiontreeclassifier, feature_names=X_train.columns)
        print(text_representation)
    i += 1
    print(score_df)
"""
















