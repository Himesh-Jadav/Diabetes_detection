# Diabetes Detection - Flask Web App 🩺

This is an End-to-End Machine Learning Project to predict whether a person has diabetes based on medical attributes.
The model is trained using the PIMA Indians Diabetes dataset and deployed as an interactive Flask web application.

# 📊 Dataset Details

Features:

Pregnancies: Number of pregnancies

Glucose: Plasma glucose concentration

BloodPressure: Diastolic blood pressure (mm Hg)

SkinThickness: Triceps skin fold thickness (mm)

Insulin: 2-Hour serum insulin (mu U/ml)

BMI: Body mass index (weight in kg/(height in m)^2)

DiabetesPedigreeFunction: Diabetes pedigree function

Age: Age (years)

Outcome: 0 = Non-diabetic, 1 = Diabetic

# Shape:

(636, 9) entries and features.

Outcome Distribution:

0 (Non-diabetic): 435 samples

1 (Diabetic): 201 samples

# ⚙️ Model Training & Evaluation

Algorithms Used:

Support Vector Machine (SVM)

Random Forest Classifier

SVM Results:

Best Parameters: {'C': 100, 'gamma': 'auto', 'kernel': 'rbf'}

Cross-Validation Accuracy: ~77.97%

Train Accuracy: ~85.80%

Test Accuracy: ~77.34%

ROC-AUC Score: 0.845

Random Forest Results:

Train Accuracy: 100%

Test Accuracy: ~80.47%

ROC-AUC Score: 0.885

# Final Model:

Random Forest selected for deployment due to better generalization on test data.

🖥️ Web Application (Flask)
Users can input their medical data via a simple HTML form.

Upon submission, the model predicts and displays:

"Diabetic" or "Non-Diabetic".
