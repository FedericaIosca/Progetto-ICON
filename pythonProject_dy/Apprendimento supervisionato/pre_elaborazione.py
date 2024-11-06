import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer

data = pd.read_csv('Dataset/diabetes.csv')

imputer = SimpleImputer(strategy='mean', missing_values=0)

data.iloc[:, :-1] = imputer.fit_transform(data.iloc[:, :-1])

features = data.drop(['Outcome'], axis=1)
labels = data['Outcome']

significant_features = ['Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction', 'Age']
features[significant_features] *= 2

scaler = MinMaxScaler()

scaler.fit(features)

features_normalized = scaler.transform(features)

train_features, test_features, train_labels, test_labels = train_test_split(features_normalized, labels, test_size=0.3, random_state=42)
train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

class_labels = np.unique(train_labels)
class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=train_labels)

class_weights_dict = dict(zip(class_labels, class_weights))

# Saving datas
np.save('Pre_Processed_Data/class_weights.npy', class_weights_dict)
np.save('Pre_Processed_Data/train_features.npy', train_features)
np.save('Pre_Processed_Data/train_labels.npy', train_labels)
np.save('Pre_Processed_Data/val_features.npy', val_features)
np.save('Pre_Processed_Data/val_labels.npy', val_labels)
np.save('Pre_Processed_Data/test_features.npy', test_features)
np.save('Pre_Processed_Data/test_labels.npy', test_labels)

print("train_features:", len(train_features))
print("test_features:", len(test_features))
print("val_features:", len(val_features))
print("total features:",len(features_normalized))