
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def show_feature_importance (classifier, X_test):
    # Bedeutung der Variablen bei der logistischen Regression anzeigen
    coefficients = classifier.coef_[0]
    feature_importance = pd.DataFrame({'Feature': X_test.columns, 'Importance': np.abs(coefficients)})
    feature_importance = feature_importance.sort_values('Importance', ascending=True)
    feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(20, 6))
    plt.show()

def show_feature_importance_tree (classifier, X_test):
    features = X_test.columns
    importances = classifier.feature_importances_
    indices = np.argsort(importances)

    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

