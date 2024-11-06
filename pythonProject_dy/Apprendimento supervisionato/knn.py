import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score

train_features = np.load('Pre_Processed_Data/train_features.npy')
train_labels = np.load('Pre_Processed_Data/train_labels.npy')
val_features = np.load('Pre_Processed_Data/val_features.npy')
val_labels = np.load('Pre_Processed_Data/val_labels.npy')
test_features = np.load('Pre_Processed_Data/test_features.npy')
test_labels = np.load('Pre_Processed_Data/test_labels.npy')

print("Before SMOTE:")
print("Class 0:", sum(train_labels == 0))
print("Class 1:", sum(train_labels == 1))

# Applica SMOTE all'insieme di addestramento
smote = SMOTE()
train_features_resampled, train_labels_resampled = smote.fit_resample(train_features, train_labels)


print("After SMOTE:")
print("Classe 0:", sum(train_labels_resampled == 0))
print("Classe 1:", sum(train_labels_resampled == 1))

#training del modello

param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]}

best_params = None
best_val_f1_score = 0

for params in param_grid['n_neighbors']:

    knn = KNeighborsClassifier(n_neighbors=params)

    knn.fit(train_features_resampled, train_labels_resampled)

    y_val_pred = knn.predict(val_features)
    val_f1_score = f1_score(val_labels, y_val_pred)

    if val_f1_score > best_val_f1_score:
        best_params = params
        best_val_f1_score = val_f1_score

print(f'Best hyperparameters: {best_params}')

#istanzazione del knn

knn = KNeighborsClassifier(n_neighbors=best_params)

knn.fit(train_features_resampled, train_labels_resampled)

#utilizzo del modello

test_predictions = knn.predict(test_features)
print("test_predictions:",test_predictions)

#valutazioni sul test

test_accuracy = accuracy_score(test_labels, test_predictions)
error_rate = 1 - test_accuracy
print("Error rate:", error_rate)

# Accuracy
test_accuracy = accuracy_score(test_labels, test_predictions)
print(f'Accuracy sul set di test: {test_accuracy:.2f}')

# Confusion Matrix
cm = confusion_matrix(test_labels, test_predictions)

# Define class labels
class_labels = ['Class 0', 'Class 1']

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=class_labels, yticklabels=class_labels)

plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

report = classification_report(test_labels, test_predictions)
print("Classification Report:\n", report)

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


#---------------------------



import numpy as np
from scipy.stats import norm


test_features = np.load('Pre_Processed_Data/test_features.npy')
test_labels = np.load('Pre_Processed_Data/test_labels.npy')
predictions = knn.predict(test_features)


accuracy = np.mean(predictions == test_labels)

num_successes = np.sum(predictions == test_labels)
num_failures = len(test_labels) - num_successes

confidence_level = 0.95  # Livello di confidenza desiderato
z = norm.ppf(1 - (1 - confidence_level) / 2)
n = len(test_labels)
p_hat = accuracy
interval_lower = (p_hat + (z**2) / (2 * n) - z * np.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * n)) / n)) / (1 + (z**2) / n)
interval_upper = (p_hat + (z**2) / (2 * n) + z * np.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * n)) / n)) / (1 + (z**2) / n)

print(f"Intervallo di confidenza ({confidence_level * 100}%): [{interval_lower:.4f}, {interval_upper:.4f}]")

scores = cross_val_score(knn, train_features, train_labels, cv=5)

mean_accuracy = scores.mean()

print("Accuracy scores for each fold:", scores)

print("Mean accuracy:", mean_accuracy)