from flask import Flask, request, jsonify
import pickle
import os
import numpy as np

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__),"..", "model", "model.pkl")

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        print("Model loaded sucessfully")

except FileNotFoundError:
    print(f"Model not found at {model_path}")
    model = None
except Exception as e:
    print(f"Error loading file: {e}")
    model=None

@app.route("/", methods=["GET"])
def home():
    return """
    <h1>This is Home Page</h1>
    <p>API is runnign sucessfully!</p>
    <p>Use POST /predict with JSON data containing "feature' array</p>
    <p>Example: {"features": [1, 2, 3, 4]}</p>
    """

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("Data received")

        if not data:
            return jsonify({"error": "No JSON data recieved"}), 400
        if "features" not in data:
            return jsonify({"error": "Missing Feature Key in JSON data",
                            "Recevied_keys": list(data.keys()) if data else []}), 400
        
        features = np.array(data['features']).reshape(1,-1)
        print(f'Features shape:', {features.shape})

        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        prediction = model.predict(features)[0]
        return jsonify({'prediction': int(prediction)})
    
    except Exception as e:
        print(f"Error in prediction {e}")
        return jsonify({'error': str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True)
