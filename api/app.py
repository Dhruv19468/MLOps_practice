from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model with better path handling
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.pkl')
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route("/")
def home():
    return """
    <h1>ML Model API</h1>
    <p>API is running successfully!</p>
    <p>Use POST /predict with JSON data containing 'features' array</p>
    <p>Example: {"features": [1, 2, 3, 4, 5]}</p>
    """

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Debug: print received data
        data = request.get_json()
        print(f"Received data: {data}")
        
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
        
        if "features" not in data:
            return jsonify({
                "error": "Missing 'features' key in JSON data",
                "received_keys": list(data.keys()) if data else []
            }), 400
        
        features = np.array(data["features"]).reshape(1, -1)
        print(f"Features shape: {features.shape}")
        
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
            
        prediction = model.predict(features)[0]
        return jsonify({"prediction": int(prediction)})
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)