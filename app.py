from flask import Flask, request, jsonify
from datetime import datetime
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model components
model = joblib.load('model.pkl')
poly = joblib.load('poly_transform.pkl')  # NEW!
start_date = joblib.load('start_date.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    date_str = data['date']
    date = pd.to_datetime(date_str)

    # Convert to number of days since training start
    days = (date - start_date).days
    days_transformed = poly.transform([[days]])  # NEW!
    
    prediction = model.predict(days_transformed)[0]
    return jsonify({'prediction': max(0, int(prediction))})

if __name__ == '__main__':
    app.run(debug=True)
