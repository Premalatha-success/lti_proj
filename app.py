from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load model
model = pickle.load(open("iris_model.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to Iris Prediction App"

# API prediction (for curl / Postman)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

# HTML Form Page
@app.route('/form')
def form():
    return render_template('upload.html')

# Handle Form Submission
@app.route('/predict_form', methods=['POST'])
def predict_form():
    sepal_length = float(request.form['sepal_length'])
    sepal_width  = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width  = float(request.form['petal_width'])

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]

    return f"Predicted Class: {prediction}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
