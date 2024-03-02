from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def determine_feature_importance(X, X_train, y_train, X_test, y_test):
    feature_names = [f"feature {i}" for i in range(X.shape[1])]
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X_train, y_train)

    # importance based on mean decrease in impurity
    start_time = time.time()
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
    forest_importances_mean_decrease_impurity = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances_mean_decrease_impurity.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    # importance based on permutation
    start_time = time.time()
    result = permutation_importance(forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
    forest_importances_permutation = pd.Series(result.importances_mean, index=feature_names)



