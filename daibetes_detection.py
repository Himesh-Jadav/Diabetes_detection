
# Diabetes Detection Model
# This notebook trains a machine learning model to predict diabetes and saves the model and preprocessing objects for a web application.

# Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

# Load Dataset
diabetes_dataset = pd.read_csv('diabetes.csv')
# Handle Missing Values
# Replace zeros with median for columns where zero is not realistic
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_with_zeros:
    diabetes_dataset[col] = diabetes_dataset[col].replace(0, diabetes_dataset[col].median())

# Remove Outliers using IQR
Q1 = diabetes_dataset.quantile(0.25)
Q3 = diabetes_dataset.quantile(0.75)
IQR = Q3 - Q1
diabetes_dataset = diabetes_dataset[~((diabetes_dataset < (Q1 - 1.5 * IQR)) | (diabetes_dataset > (Q3 + 1.5 * IQR))).any(axis=1)]

# Inspect Dataset
print("Dataset Head:")
print(diabetes_dataset.head())
print("\nDataset Shape:", diabetes_dataset.shape)
print("\nDataset Description:")
print(diabetes_dataset.describe())
print("\nOutcome Distribution:")
print(diabetes_dataset['Outcome'].value_counts())

# Separate Features and Target
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Feature Engineering: Add Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X = poly.fit_transform(X)

# Standardize Features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Feature Selection using RFE
estimator = RandomForestClassifier(random_state=2)
selector = RFE(estimator, n_features_to_select=5, step=1)
selector = selector.fit(X, Y)
X = selector.transform(X)

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Address Class Imbalance with SMOTE
smote = SMOTE(random_state=2)
X_train, Y_train = smote.fit_resample(X_train, Y_train)

# Train SVM with Hyperparameter Tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto', 0.1, 0.01]
}
classifier = SVC(probability=True)
grid = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, Y_train)
classifier = grid.best_estimator_
print('Best SVM Parameters:', grid.best_params_)
print('Best SVM Cross-Validation Accuracy:', grid.best_score_)

# Evaluate SVM
svm_train_pred = classifier.predict(X_train)
svm_test_pred = classifier.predict(X_test)
print('SVM Train Accuracy:', accuracy_score(svm_train_pred, Y_train))
print('SVM Test Accuracy:', accuracy_score(svm_test_pred, Y_test))
print('\nSVM Classification Report:')
print(classification_report(Y_test, svm_test_pred))
print('SVM ROC-AUC:', roc_auc_score(Y_test, classifier.predict_proba(X_test)[:, 1]))

# Train Random Forest as Alternative Model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=2)
rf_classifier.fit(X_train, Y_train)
rf_train_pred = rf_classifier.predict(X_train)
rf_test_pred = rf_classifier.predict(X_test)
print('\nRandom Forest Train Accuracy:', accuracy_score(rf_train_pred, Y_train))
print('Random Forest Test Accuracy:', accuracy_score(rf_test_pred, Y_test))
print('\nRandom Forest Classification Report:')
print(classification_report(Y_test, rf_test_pred))
print('Random Forest ROC-AUC:', roc_auc_score(Y_test, rf_classifier.predict_proba(X_test)[:, 1]))

# Cross-Validation for Final Model
final_classifier = rf_classifier if accuracy_score(rf_test_pred, Y_test) > accuracy_score(svm_test_pred, Y_test) else classifier
scores = cross_val_score(final_classifier, X, Y, cv=5, scoring='accuracy')
print('\nFinal Model Cross-Validation Accuracy:', scores.mean(), '±', scores.std())

# Save Model and Preprocessing Objects
joblib.dump(final_classifier, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(poly, 'poly.pkl')
joblib.dump(selector, 'selector.pkl')
print('\nModel and preprocessing objects saved as diabetes_model.pkl, scaler.pkl, poly.pkl, and selector.pkl')

# Make Prediction for a Sample Input
input_data = (4, 110, 92, 0, 0, 37.6, 0.191, 30)
data_as_arr = np.array(input_data).reshape(1, -1)
data_as_arr = poly.transform(data_as_arr)
std_data = scaler.transform(data_as_arr)
std_data = selector.transform(std_data)
prediction = final_classifier.predict(std_data)
print('\nPrediction for Input:', input_data)
print('Result:', 'Non-Diabetic (Stay Fit)' if prediction[0] == 0 else 'Diabetic (Be Careful)')