
import data_load as dl
import prepare_data as prepdata
import feature_importance as fi
import parameter_tuning as pt
import feature_selection as fs

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
y = prepared_df.success

# für SVM und die logistische Regression nur die numerischen Spalten behalten
X_num = X.select_dtypes(include=np.number)
X_num.to_csv('X_numerische_Datengrundlage.csv', sep=';')

X = prepared_df.drop(columns='success', axis=1)

# für Decision Tree, Random Forest, XGBoost die One-Hot-Encoding Spalten entfernen
"""
X_combined = X.drop(columns=['weekday_Monday', 'weekday_Tuesday', 'weekday_Wednesday', 'weekday_Thursday', 'weekday_Friday', 'weekday_Saturday',
                             'weekday_Sunday', 'month_January', 'month_February', 'PSP_Moneycard', 'PSP_Goldcard', 'PSP_UK_Card',
                             'PSP_Simplecard', 'card_Visa', 'card_Master', 'card_Diners', 'country_Germany', 'country_Austria', 'country_Switzerland', 'tmsp'],axis=1)

X_combined.to_csv('X_kombinierte_Datengrundlage.csv', sep=';')
"""



# Unterteilen in Test- und Trainingsmenge
# für Random Forest und co
# X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(X_combined, y, test_size=0.2, random_state=42)
# für SVM und logistische Regression
X_train_num, X_test_num, y_train_num, y_test_num = train_test_split(X_num, y, test_size=0.2, random_state=42)


# feature Selection für kombinierte Verfahren
# X_train_relevant_combined = fs.feature_selection_TreeClassifier(X_train_combined, y_train_combined, "mean")
# X_train_relevant_combined.to_csv('X_train_relevant_combined.csv', sep=';')

# feature Selection für numerische Verfahren --> Vorher Encoden
encoder = LabelEncoder()
categorical_features = X_train_num.columns.tolist()
for each in categorical_features:
    X_train_num[each] = encoder.fit_transform(X_train_num[each])

# X_train_relevant_num = fs.feature_selection_TreeClassifier_SelectFromModel(X_train_num, y_train_num, 'mean')
# X_train_relevant_num.to_csv('X_train_relevant_num.csv', sep=';')

X_train_relevant_num_kbest = fs.feature_selection_TreeClassifier_KBest(X_train_num, y_train_num, 10)
X_train_relevant_num_kbest.to_csv('X_train_relevant_num_kbest.csv', sep=';')

X_test_relevant_num_kbest = X_test_num[X_train_relevant_num_kbest.columns]


best_logistic = pt.logistic_regression_tuning(X_train_relevant_num_kbest,  y_train_num,  target)
print('Results logistic regression')
logistic_regression_result = pt.evaluate_model(best_logistic, X_train_relevant_num_kbest, X_test_relevant_num_kbest, y_train_num, y_test_num)
fi.show_feature_importance(best_logistic, X_test_relevant_num_kbest)


best_decision_tree = pt.decision_tree_tuning(X_train_relevant_num_kbest,  y_train_num,  target)
print('Results decision tree')
decision_tree_result = pt.evaluate_model(best_decision_tree, X_train_relevant_num_kbest, X_test_relevant_num_kbest, y_train_num, y_test_num)
fi.show_feature_importance_tree(best_decision_tree, X_test_relevant_num_kbest)


best_random_forest = pt.random_forest_tuning(X_train_relevant_num_kbest,  y_train_num,  target)
print('Results random forest')
random_forest_result = pt.evaluate_model(best_random_forest,X_train_relevant_num_kbest, X_test_relevant_num_kbest, y_train_num, y_test_num)
fi.show_feature_importance_tree(best_random_forest, X_test_relevant_num_kbest)


best_extra_tree = pt.extra_trees_tuning(X_train_relevant_num_kbest,  y_train_num,  target)
print('Results extra tree')
random_extra_tree = pt.evaluate_model(best_extra_tree, X_train_relevant_num_kbest, X_test_relevant_num_kbest, y_train_num, y_test_num)
fi.show_feature_importance_tree(best_extra_tree, X_test_relevant_num_kbest)



best_xg = pt.xgboost_tuning(X_train_relevant_num_kbest,  y_train_num,  target)
print('Results XGBoost')
xg_result = pt.evaluate_model(best_xg, X_train_relevant_num_kbest, X_test_relevant_num_kbest, y_train_num, y_test_num)
fi.show_feature_importance_tree(best_xg, X_test_relevant_num_kbest)


# Scaling the features using pipeline
pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])
scaled_X_train = pipeline.fit_transform(X_train_relevant_num_kbest)
scaled_X_test = pipeline.fit_transform(X_test_relevant_num_kbest)


best_svm = pt.svm_tuning(scaled_X_train, y_train_num, target)
print('Results SVM')
svm_result = pt.evaluate_model(best_svm, scaled_X_train, scaled_X_test, y_train_num, y_test_num)
fi.show_feature_importance(best_svm, X_test_relevant_num_kbest )











