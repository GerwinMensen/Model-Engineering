import EDA
import data_load as dl
import prepare_data as prepdata
import feature_importance as fi
import parameter_tuning as pt
import feature_selection as fs
import EDA as eda

# Import relevanter Packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Import relevanter sklearn Packages
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Variablen definieren
filename_credit_transactions = r'PSP_Jan_Feb_2019.xlsx'
# target = "roc_auc"
target = "f1"
# target = "accuracy"

# entweder die folgenden beiden Befehle
# Daten einlesen
# unprepared_df = dl.data_load(filename=filename_credit_transactions)
# Daten aufbereiten
# prepared_df = prepdata.prepare_data(unprepared_df)




# oder diese beiden Befehle
prepared_df = dl.data_load(filename='Datengrundlage.csv')
prepared_df = prepared_df.drop(columns='Unnamed: 0', axis=1)






# Unterteilen der vorbereiten Daten nach abh. und unabhängiger Variablen
# die Erfolgswahrscheinlichkeit soll prognostiziert werden --> Spalte success
X = prepared_df.drop(columns='success', axis=1)
y = prepared_df.success



# EDA.make_EDA(X, y)

# die Spalten success_fee und month_february entfernen, da diese durch month_january bzw. fail_fee erklärt werden können
X = X.drop(columns=['success_fee', 'month_February', 'has_predecessors'], axis=1)

# nur die numerischen Spalten behalten
X_num = X.select_dtypes(include=np.number)

X_num.to_csv('X_numerische_Datengrundlage.csv', sep=';')

# Lösche alle Spalten, deren Namen mit "weekday" beginnen
columns_to_drop = [col for col in X_num.columns if col.startswith(('weekday','month','PSP','country','card'))]
X_trees = X_num.drop(columns=columns_to_drop)




# Unterteilen in Test- und Trainingsmenge
X_train, X_test, y_train, y_test = train_test_split(X_trees, y, test_size=0.2, random_state=42)

# feature Selection mit KBest und dem F-Wert (10 besten Attribute bleiben erhalten)
X_train_relevant = fs.feature_selection_TreeClassifier_KBest(X_train, y_train, 10)
X_train_relevant.to_csv('X_train_relevant_trees.csv', sep=';')

X_test_relevant = X_test[X_train_relevant.columns]





# baseline model
baseline_model= pt.baseline_model(X_train_relevant, y_train, X_test_relevant, y_test)
print('Results baseline model')
baseline_model_result = pt.evaluate_model(baseline_model, X_train_relevant, X_test_relevant, y_train, y_test)



best_decision_tree = pt.decision_tree_tuning(X_train_relevant,  y_train,  target)
print('Results decision tree')
decision_tree_result = pt.evaluate_model(best_decision_tree, X_train_relevant, X_test_relevant, y_train, y_test)
fi.show_feature_importance_tree(best_decision_tree, X_test_relevant)


best_random_forest = pt.random_forest_tuning(X_train_relevant,  y_train,  target)
print('Results random forest')
random_forest_result = pt.evaluate_model(best_random_forest,X_train_relevant, X_test_relevant, y_train, y_test)
fi.show_feature_importance_tree(best_random_forest, X_test_relevant)


best_extra_tree = pt.extra_trees_tuning(X_train_relevant,  y_train,  target)
print('Results extra tree')
random_extra_tree = pt.evaluate_model(best_extra_tree, X_train_relevant, X_test_relevant, y_train, y_test)
fi.show_feature_importance_tree(best_extra_tree, X_test_relevant)



best_xg = pt.xgboost_tuning(X_train_relevant,  y_train,  target)
print('Results XGBoost')
xg_result = pt.evaluate_model(best_xg, X_train_relevant, X_test_relevant, y_train, y_test)
fi.show_feature_importance_tree(best_xg, X_test_relevant)






# Lösche alle Spalten, deren Namen mit encoded beginnen
columns_to_drop = [col for col in X_num.columns if col.startswith('encoded')]
X_mathematical = X_num.drop(columns=columns_to_drop)







# Unterteilen in Test- und Trainingsmenge
X_train, X_test, y_train, y_test = train_test_split(X_mathematical, y, test_size=0.2, random_state=42)
# feature Selection mit KBest und dem F-Wert (10 besten Attribute bleiben erhalten)
X_train_relevant = fs.feature_selection_TreeClassifier_KBest(X_train, y_train, 10)
X_train_relevant.to_csv('X_train_relevant_mathematical.csv', sep=';')
X_test_relevant = X_test[X_train_relevant.columns]


best_logistic = pt.logistic_regression_tuning(X_train_relevant,  y_train,  target)
print('Results logistic regression')
logistic_regression_result = pt.evaluate_model(best_logistic, X_train_relevant, X_test_relevant, y_train, y_test)
fi.show_feature_importance(best_logistic, X_test_relevant)


# Scaling the features using pipeline
pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])
scaled_X_train = pipeline.fit_transform(X_train_relevant)
scaled_X_test = pipeline.fit_transform(X_test_relevant)


best_svm = pt.svm_tuning(scaled_X_train, y_train, target)
print('Results SVM')
svm_result = pt.evaluate_model(best_svm, scaled_X_train, scaled_X_test, y_train, y_test)
fi.show_feature_importance(best_svm, X_test_relevant )











