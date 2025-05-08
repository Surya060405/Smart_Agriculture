from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model once
model = joblib.load('model_pipeline.pkl')  # Ensure this file is in the same directory

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    temp = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    crop = request.form['crop']

    # Prepare input for prediction
    input_df = pd.DataFrame([[temp, humidity, crop]], columns=['Temperature(°C)', 'Humidity(%)', 'Crop'])

    # Predict
    prediction = model.predict(input_df)[0]
    prediction = round(prediction, 2)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
