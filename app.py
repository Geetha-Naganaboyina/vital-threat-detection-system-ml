from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

# Global list to store prediction history
prediction_history = []

# Load model and metrics
MODEL_PATH = 'models/health_model.pkl'
METRICS_PATH = 'models/metrics.pkl'

model = None
metrics = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
if os.path.exists(METRICS_PATH):
    metrics = joblib.load(METRICS_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not trained yet'}), 500
    
    try:
        data = request.json
        hr = float(data.get('heart_rate'))
        spo2 = float(data.get('spo2'))
        temp = float(data.get('temperature'))
        steps = float(data.get('steps'))
        
        prediction = model.predict([[hr, spo2, temp, steps]])[0]
        result = "Health Risk Detected" if prediction == 1 else "Normal Condition"
        alert_class = "alert-danger" if prediction == 1 else "alert-success"
        
        prediction_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'heart_rate': hr,
            'spo2': spo2,
            'temperature': temp,
            'steps': steps,
            'result': result,
            'alert_class': alert_class
        }
        
        # Add to history (keep only the latest 50 for memory safety)
        prediction_history.insert(0, prediction_entry)
        if len(prediction_history) > 50:
            prediction_history.pop()
        
        return jsonify({
            'result': result,
            'alert_class': alert_class
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', metrics=metrics, history=prediction_history)

@app.route('/api/metrics')
def get_metrics():
    return jsonify(metrics)

if __name__ == '__main__':
    app.run(debug=True)
