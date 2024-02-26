# import python scripts
import data_load as dl
import prepare_data as prepdata


# import packages
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt

# Variablen definieren
filename_credit_transactions = r'PSP_Jan_Feb_2019.xlsx'

# Daten einlesen
# unprepared_df = dl.data_load(filename=filename_credit_transactions)
# Daten aufbereiten
# prepared_df = prepdata.prepare_data(unprepared_df)

prepared_df = dl.data_load(filename='Datengrundlage.csv')
prepared_df = prepared_df.drop(columns='Unnamed: 0', axis=1)

print(prepared_df)
# Unterteilen der vorbereiten Daten nach abh. und unabhängiger Variablen
# Zunächst soll die Erfolgswahrscheinlichkeit prognostiziert werden --> Spalte success
X = prepared_df.drop(columns='success', axis=1)

# nur die numerischen Spalten behalten
X = X.select_dtypes(include=np.number)
y = prepared_df.success

print(X)
print(y)

# SVM durchführen
# clf = svm.SVC()
# clf.fit(x, y)

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
# Logistische Regression durchführen
# model_logistic_regression = LogisticRegression(random_state=0).fit(X_scaled, y)
model_logistic_regression = LogisticRegression()
model_logistic_regression.fit(X_scaled, y)

# Ergebnis logistische Regression
print("Ergebnis logistische Regression")
predicted_logistic_regression = model_logistic_regression.predict(X_scaled)
print(np.sum(predicted_logistic_regression * 1 == y))
print(model_logistic_regression.predict_proba(X_scaled))
predictedProbs = model_logistic_regression.predict_proba(X_scaled)[:,1]



print("Likelihood-Wert")
print(np.prod(predictedProbs * y + (1 - predictedProbs) * (1-y)))


print(np.sum(predicted_logistic_regression * 1 == y) / len(X_scaled))


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