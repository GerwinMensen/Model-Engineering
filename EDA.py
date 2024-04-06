import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def make_EDA (X,y):

    # Lösche alle Spalten, deren Namen mit "weekday, month, PSP, country, card, day_, hour_, minute_ oder second_" beginnen
    columns_to_drop = [col for col in X.columns if col.startswith(('day_', 'hour_', 'minute_', 'second_'))]
    X = X.drop(columns=columns_to_drop)

    # Grundlegende Statistiken für numerische Features
    # print(X.describe())
    print(y.describe())

    # Überprüfen auf fehlende Werte
    # print(X.isnull().sum())
    print(y.isnull().sum())

    num_and_binary_columns = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    num_and_binary_features = X[num_and_binary_columns]

    # binäre Variablen bestimmen
    binary_columns = num_and_binary_features.columns[(num_and_binary_features.isin([0, 1]).all())]
    binary_features = num_and_binary_features[binary_columns]

    # numerische Variablen bestimmen
    num_columns = [col for col in num_and_binary_features.columns if col not in binary_features.columns]
    num_features = num_and_binary_features[num_columns]

    # Verteilungen der numerischen Variablen
    X[num_columns].hist(bins=60, figsize=(14, 10), layout=(3, -1))
    plt.suptitle('Histogramme numerischer Merkmale')
    plt.savefig('Histogram numerischer Merkmale.png')
    plt.show()


    # Verteilungen der binären Variablen
    bins = [-0.5, 0.5, 1.5]
    X[binary_columns].hist(bins=[-0.5, 0.5, 1.5], layout=(4, -1), figsize=(22, 16))
    # X[binary_columns].hist(bins=[0,0.5,1], figsize=(20, 15), layout=(3, -1), ec="k")
    plt.xticks([0, 1], ['0', '1'])
    plt.suptitle('Histogramme binärer Merkmale')
    plt.savefig('Histogram binärer Merkmale.png')
    plt.show()



    # Korrelationen zwischen den numerischen Features
    plt.figure(figsize=(25, 20))
    sns.heatmap(X[num_and_binary_columns].corr(),  fmt=".2f", cmap='coolwarm')
    plt.title('Korrelationsmatrix der Merkmale')
    plt.savefig('Korrelationsmatrix der Merkmale.png')
    plt.show()

    # Beziehungen zwischen Features und der Zielvariablen visualisieren
    # Beispiel: Beziehung zwischen 'amount' und 'success'
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='success', y='amount', data=X.join(y))
    plt.title('Beziehung zwischen amount und success')
    plt.show()



    # Beziehungen zwischen Features und der Zielvariablen visualisieren
    # Beispiel: Beziehung zwischen 'hour' und 'success'
    plt.figure(figsize=(6, 4))
    sns.lineplot(x='hour', y='success', data=X.join(y))
    plt.title('Beziehung zwischen hour und success')
    plt.show()

    # Beziehungen zwischen Features und der Zielvariablen visualisieren
    # Beispiel: Beziehung zwischen 'second' und 'success'
    plt.figure(figsize=(6, 4))
    sns.lineplot(x='second', y='success', data=X.join(y))
    plt.title('Beziehung zwischen second und success')
    plt.show()

    # Beziehungen zwischen Features und der Zielvariablen visualisieren
    # Beispiel: Beziehung zwischen 'day' und 'success'
    plt.figure(figsize=(6, 4))
    sns.lineplot(x='day', y='success', data=X.join(y) ) # , ci=None)
    plt.title('Beziehung zwischen day und success')
    plt.show()