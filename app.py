from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        try:
            # Get form data
            features = [
                float(request.form['amount']),
                float(request.form['time']),
                float(request.form['location']),
                float(request.form['transaction_type']),
                float(request.form['balance']),
                float(request.form['frequency'])
            ]
            
            # Scale and predict
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)[0]
            result = "Fraudulent" if prediction == 1 else "Legitimate"
            
        except Exception as e:
            result = f"Error: {str(e)}"
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)