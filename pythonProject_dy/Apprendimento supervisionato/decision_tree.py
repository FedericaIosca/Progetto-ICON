#Decision Tree

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, algorithms
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from deap import base, creator, tools, algorithms
from sklearn.metrics import f1_score
import numpy as np
from scipy.stats import norm


train_features = np.load('Pre_Processed_Data/train_features.npy')
train_labels = np.load('Pre_Processed_Data/train_labels.npy')
val_features = np.load('Pre_Processed_Data/val_features.npy')
val_labels = np.load('Pre_Processed_Data/val_labels.npy')
test_features = np.load('Pre_Processed_Data/test_features.npy')
test_labels = np.load('Pre_Processed_Data/test_labels.npy')
class_weights_dict = np.load('Pre_Processed_Data/class_weights.npy', allow_pickle=True).item()


#Funzione di fitness da massimizzare e creazione di tipi di individui e popolazioni per l'algoritmo genetico

def fitness(individual):

    max_depth, min_samples_split = individual

    dt = DecisionTreeClassifier(class_weight=class_weights_dict,max_depth=max_depth, min_samples_split=min_samples_split)

    dt.fit(train_features, train_labels)

    val_predictions = dt.predict(val_features)
    val_f1_score = f1_score(val_labels, val_predictions)

    return val_f1_score,


if not hasattr(creator, 'FitnessMax'):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, 'Individual'):
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("max_depth", np.random.randint, 1, 10)
toolbox.register("min_samples_split", np.random.randint, 2, 10)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.max_depth, toolbox.min_samples_split), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=[1, 2], up=[10, 10], indpb=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)

#Creazione di una popolazione di partenza casuale

population = toolbox.population(n=50)

#Esecuzione dell'algoritmo genetico per trovare il miglior iperparametro

result = algorithms.eaSimple(population, toolbox,
                             cxpb=0.5, mutpb=0.2,
                             ngen=10, verbose=False)

#Estrazione del miglior iperparametro con l'algoritmo genetico

best_individual = tools.selBest(population, k=1)[0]
best_params = {'max_depth': best_individual[0], 'min_samples_split': best_individual[1]}
print(f'Migliori hyperparameters: {best_params}')

#Istanziazione del decision tree usando il miglior iperparametro trovato attraverso l'algoritmo genetico

dt = DecisionTreeClassifier(class_weight=class_weights_dict, max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'])


#Training del modello con il training set

dt.fit(train_features, train_labels)
DecisionTreeClassifier(class_weight={0: 0.7660714285714286,
                                     1: 1.4395973154362416},
                       max_depth=4, min_samples_split=6)

#Usiamo il modello trainato per effettuare predizioni sul test set

test_predictions = dt.predict(test_features)
print(test_predictions)

#Valutazioni delle prestazioni del modello sul test set

test_accuracy = accuracy_score(test_labels, test_predictions)
print(f'Accuracy sul set di test: {test_accuracy:.2f}')

#Matrice di confusione

cm = confusion_matrix(test_labels, test_predictions)

class_labels = ['Class 0', 'Class 1']

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=class_labels, yticklabels=class_labels)

plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

#Classification report

report = classification_report(test_labels, test_predictions)
print("Classification Report:\n", report)

#ROC and AUC

fpr, tpr, thresholds = roc_curve(test_labels, test_predictions)
auc = roc_auc_score(test_labels, test_predictions)

print("AUC:", auc)

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

#Intervallo di confidenza

predictions = dt.predict(test_features)

# accuracy
accuracy = np.mean(predictions == test_labels)

# numero di successi e fallimenti
num_successes = np.sum(predictions == test_labels)
num_failures = len(test_labels) - num_successes

confidence_level = 0.95
z = norm.ppf(1 - (1 - confidence_level) / 2)
n = len(test_labels)
p_hat = accuracy
interval_lower = (p_hat + (z**2) / (2 * n) - z * np.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * n)) / n)) / (1 + (z**2) / n)
interval_upper = (p_hat + (z**2) / (2 * n) + z * np.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * n)) / n)) / (1 + (z**2) / n)

print(f"Intervallo di confidenza ({confidence_level * 100}%): [{interval_lower:.4f}, {interval_upper:.4f}]")

#Cross validation

scores = cross_val_score(dt, train_features, train_labels, cv=5)

# mean accuracy
mean_accuracy = scores.mean()

print("Accuracy scores for each fold:", scores)
print("Mean accuracy:", mean_accuracy)



#RANDOM FOREST


#Definizione di una funzione di fitness da massimizzare e creazione di tipi di individui e popolazioni per l'algoritmo genetico

def fitness(individual):
    n_estimators, max_depth, min_samples_split = individual

    rf = RandomForestClassifier(class_weight=class_weights_dict, n_estimators=n_estimators, max_depth=max_depth,
                                min_samples_split=min_samples_split)

    rf.fit(train_features, train_labels)

    y_val_pred = rf.predict(val_features)
    val_f1_score = f1_score(val_labels, y_val_pred)

    return val_f1_score,


if not hasattr(creator, 'FitnessMax'):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, 'Individual'):
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("n_estimators", np.random.randint, 10, 200)
toolbox.register("max_depth", np.random.randint, 1, 10)
toolbox.register("min_samples_split", np.random.randint, 2, 10)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.n_estimators, toolbox.max_depth, toolbox.min_samples_split), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Registra gli operatori genetici da utilizzare dall'algoritmo genetico
toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=[10, 1, 2], up=[200, 10, 10], indpb=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)

#Creazione popolazione di partenza

population = toolbox.population(n=50)

#Esecuzione dell'algoritmo genetico per trovare il miglior iperparametro

result = algorithms.eaSimple(population, toolbox,
                             cxpb=0.5, mutpb=0.2,
                             ngen=10, verbose=False)

#Estrazione del miglior iperparametro dall'algoritmo genetico

best_individual = tools.selBest(population, k=1)[0]
best_params = {'n_estimators': best_individual[0], 'max_depth': best_individual[1], 'min_samples_split': best_individual[2]}
print(f'Best hyperparameters: {best_params}')

#Istanziazione del Random Forest utilizzando i migliori iperparametri trovati dall'algoritmo genetico

rf = RandomForestClassifier(class_weight=class_weights_dict, n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'])

#Training del modello con il training set

rf.fit(train_features, train_labels)
RandomForestClassifier(class_weight={0: 0.7660714285714286,
                                     1: 1.4395973154362416},
                       max_depth=6, min_samples_split=7, n_estimators=50)

#Uso del modello trainato per effettuare predizioni sul test set

y_test_pred = rf.predict(test_features)

#Valutazione delle performance sul test set

# Accuracy
test_accuracy = accuracy_score(test_labels, y_test_pred)
print(f'Accuracy sul set di test: {test_accuracy:.2f}')

#Matrice di confusione

cm = confusion_matrix(test_labels, y_test_pred)

class_labels = ['Class 0', 'Class 1']

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=class_labels, yticklabels=class_labels)

plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

#Classification report

report = classification_report(test_labels, y_test_pred)
print("Classification Report:\n", report)

#ROC and AUC

fpr, tpr, thresholds = roc_curve(test_labels, y_test_pred)
auc = roc_auc_score(test_labels, y_test_pred)

print("AUC:", auc)

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

#Intervallo di confidenza

predictions = rf.predict(test_features)

accuracy = np.mean(predictions == test_labels)

num_successes = np.sum(predictions == test_labels)
num_failures = len(test_labels) - num_successes

confidence_level = 0.95
z = norm.ppf(1 - (1 - confidence_level) / 2)
n = len(test_labels)
p_hat = accuracy
interval_lower = (p_hat + (z**2) / (2 * n) - z * np.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * n)) / n)) / (1 + (z**2) / n)
interval_upper = (p_hat + (z**2) / (2 * n) + z * np.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * n)) / n)) / (1 + (z**2) / n)

print(f"Confidence interval ({confidence_level * 100}%): [{interval_lower:.4f}, {interval_upper:.4f}]")

#Cross validation

scores = cross_val_score(rf, train_features, train_labels, cv=5)

mean_accuracy = scores.mean()

print("Accuracy scores for each fold:", scores)

print("Mean accuracy:", mean_accuracy)


