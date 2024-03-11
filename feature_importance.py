from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def determine_feature_importance(X, X_train, y_train, X_test, y_test):
    feature_names = [f"{col}" for col in X.columns]
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X_train, y_train)

    # importance based on mean decrease in impurity
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    forest_importances_mean_decrease_impurity = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances_mean_decrease_impurity.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()

    # importance based on permutation
    result = permutation_importance(forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    forest_importances_permutation = pd.Series(result.importances_mean, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances_permutation.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using permutation")
    ax.set_ylabel("permutation")
    fig.tight_layout()
    plt.show()
    result_df = pd.DataFrame(columns=['Mean decrease impurity', 'Permutation'])
    result_df['Mean decrease impurity'] = forest_importances_mean_decrease_impurity
    result_df['Permutation'] = forest_importances_permutation
    return result_df.transpose()


# Bedeutung der Variablen bei SVM berechnen
def f_importances(coef, feature_names):
    imp = coef
    imp,feature_names = zip(*sorted(zip(imp,feature_names)))
    plt.barh(range(len(feature_names)), imp, align='center')
    plt.yticks(range(len(feature_names)), feature_names)
    plt.show()

#    https://stackoverflow.com/questions/41592661/determining-the-most-contributing-features-for-svm-classifier-in-sklearn