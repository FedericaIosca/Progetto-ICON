#Analisi del dataset

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scipy import stats

import numpy as np

#Controllo se il dataset è sbilanciato

data = pd.read_csv('Dataset/diabetes.csv')

class_counts = data['Outcome'].value_counts()

plt.bar(class_counts.index, class_counts.values)
plt.xticks(class_counts.index)
plt.xlabel('Class')
plt.ylabel('N. of examples')
plt.show()

#Controllo dei valori nulli
print("\nControllo valori nulli\n")
print(data.isnull().sum())

#Matrice di correlazione

corr_matrix = data.corr()

pd.set_option('display.max_columns', None)
print("\nMatrice di correlazione\n")
print(corr_matrix)

sns.heatmap(corr_matrix, annot=True)
plt.show()

#Verifica della media glicemica per ogni classe

glucose_mean = data.groupby('Outcome')['Glucose'].mean()

ax = glucose_mean.plot(kind='bar')
plt.xlabel('Outcome')
plt.ylabel('Media dello zucchero nel sangue')

for i in ax.containers:
    ax.bar_label(i)

plt.title("Media glicemica")

plt.show()

#T-student test

print(f"\n T-Student: \n")

features = data.drop(['Outcome'], axis=1)
labels = data['Outcome']

class_0_data = features[labels == 0]
class_1_data = features[labels == 1]

for feature in features.columns:
    mean_0 = class_0_data[feature].mean()
    mean_1 = class_1_data[feature].mean()

    std_0 = class_0_data[feature].std()
    std_1 = class_1_data[feature].std()

    n_0 = len(class_0_data)
    n_1 = len(class_1_data)

    t_score = (mean_1 - mean_0) / np.sqrt((std_1 ** 2 / n_1) + (std_0 ** 2 / n_0))

    df = n_0 + n_1 - 2

    p_value = stats.t.sf(np.abs(t_score), df) * 2



    print(f"Feature: {feature}")
    print(f"t-score: {t_score}")
    print(f"p-value: {p_value}")
    print("----------------------------------------------------------------")

