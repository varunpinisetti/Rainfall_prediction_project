from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and preprocessors
model = pickle.load(open("model/rainfall_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
imputer = pickle.load(open("model/imputer.pkl", "rb"))

# Numeric features ONLY (matches training)
FEATURES = [
    'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
    'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
    'Humidity9am', 'Humidity3pm',
    'Pressure9am', 'Pressure3pm',
    'Cloud9am', 'Cloud3pm',
    'Temp9am', 'Temp3pm'
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.form.to_dict()

    # Build input row safely
    row = []
    for f in FEATURES:
        value = input_data.get(f, 0)
        row.append(float(value) if value != "" else 0)

    df = pd.DataFrame([row], columns=FEATURES)

    # Preprocess
    df = imputer.transform(df)
    df = scaler.transform(df)

    # Predict
    pred = model.predict(df)[0]

    if pred == 1:
        return render_template("chance.html")
    else:
        return render_template("noChance.html")

if __name__ == "__main__":
    app.run(debug=True)
