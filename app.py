from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and preprocessing objects
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')
poly = joblib.load('poly.pkl')
selector = joblib.load('selector.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        input_data = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]

        # Check for negative values
        input_names = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
                       'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age']
        for value, name in zip(input_data, input_names):
            if value < 0:
                return render_template('index.html', prediction_text=f'Error: {name} cannot be negative.')

        # Preprocess input
        input_array = np.array(input_data).reshape(1, -1)
        input_poly = poly.transform(input_array)
        input_scaled = scaler.transform(input_poly)
        input_selected = selector.transform(input_scaled)

        # Make prediction
        prediction = model.predict(input_selected)
        result = 'Non-Diabetic (Stay Fit)' if prediction[0] == 0 else 'Diabetic (Be Careful)'
        return render_template('index.html', prediction_text=f'Prediction: {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: Invalid input. Please enter valid numbers.')

if __name__ == '__main__':
    app.run(debug=True)