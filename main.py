
import data_load as dl
import prepare_data as prepdata
import remove_collinearity as rc
import feature_importance as fi


# import packages
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer


names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
#    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]


classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
#    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    AdaBoostClassifier(algorithm="SAMME", random_state=42),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]





# Variablen definieren
filename_credit_transactions = r'PSP_Jan_Feb_2019.xlsx'

# Daten einlesen
# unprepared_df = dl.data_load(filename=filename_credit_transactions)
# Daten aufbereiten
# prepared_df = prepdata.prepare_data(unprepared_df)

prepared_df = dl.data_load(filename='Datengrundlage.csv')
prepared_df = prepared_df.drop(columns='Unnamed: 0', axis=1)

# Unterteilen der vorbereiten Daten nach abh. und unabhängiger Variablen
# die Erfolgswahrscheinlichkeit soll prognostiziert werden --> Spalte success
X = prepared_df.drop(columns='success', axis=1)


# nur die numerischen Spalten behalten
X = X.select_dtypes(include=np.number)
# X = rc.calculate_vif_(X, 5)
X.to_csv('X.csv', sep=';')
y = prepared_df.success

# Unterteilen in Test- und Trainingsmenge
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

fi.determine_feature_importance(X, X_train, y_train, X_test, y_test)

# über Klassifizierungsmethoden iterieren
for name, clf in zip(names, classifiers):

    clf = make_pipeline(StandardScaler(), clf)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    # Bedeutung der Variablen bei der logistischen Regression anzeigen
    # coefficients = clf.coef_[0]
    # feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(coefficients)})
    # feature_importance = feature_importance.sort_values('Importance', ascending=True)
    # feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))
    # plt.show()

    """
    DecisionBoundaryDisplay.from_estimator(
        clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
    )
    
    
    # Plot the training points
    ax.scatter(
        X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
    )
    # Plot the testing points
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap=cm_bright,
        edgecolors="k",
        alpha=0.6,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    # if ds_cnt == 0:
    ax.set_title(name)
    ax.text(
        x_max - 0.3,
        y_min + 0.3,
        ("%.2f" % score).lstrip("0"),
        size=15,
        horizontalalignment="right",
    )
    
    i += 1
    """

# SVM durchführen
# clf = svm.SVC()
# clf.fit(x, y)

"""
# Logistische Regression durchführen
# model_logistic_regression = LogisticRegression(random_state=0).fit(X_scaled, y)
model_logistic_regression = LogisticRegression(max_iter=10000)
model_logistic_regression.fit(X_train, y_train)

# Ergebnis logistische Regression
print("Ergebnis logistische Regression")
predicted_logistic_regression = model_logistic_regression.predict(X_test)
print(np.sum(predicted_logistic_regression * 1 == y_test))
print(model_logistic_regression.predict_proba(X_test))
predictedProbs = model_logistic_regression.predict_proba(X_test)[:,1]

print("Likelihood-Wert")
print(np.prod(predictedProbs * y_test + (1 - predictedProbs) * (1-y_test)))

print("Genauigkeit")
print(np.sum(predicted_logistic_regression * 1 == y_test) / len(X_test))

# Bedeutung der Variablen bei der logistischen Regression anzeigen
coefficients = model_logistic_regression.coef_[0]
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(coefficients)})
feature_importance = feature_importance.sort_values('Importance', ascending=True)
feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))
plt.show()
"""

"""
# Bedeutung der Variablen bei SVM berechnen
def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()

    https://stackoverflow.com/questions/41592661/determining-the-most-contributing-features-for-svm-classifier-in-sklearn
"""
