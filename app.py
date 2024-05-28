from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
knn = joblib.load('knn_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    new_rssi = data['rssi_value']
    longitude = data['longitude']
    latitude = data['latitude']
    new_data = pd.DataFrame([[new_rssi, longitude, latitude]], columns=['RSSI', 'X', 'Y'])
    
    # Make prediction
    predicted_room = knn.predict(new_data)[0]
    
    return jsonify({'predicted_room': predicted_room})

if __name__ == '__main__':
    app.run(debug=True)
