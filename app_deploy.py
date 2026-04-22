from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import warnings
import time
import threading

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "maternal_health_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "maternal_health_scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE_DIR, "maternal_health_labels.pkl"), "rb") as f:
    le = pickle.load(f)

state = {
    "temp": None,
    "bpm": None,
    "stress": None,
    "spo2": None,
    "kicks": None,
    "timestamp": "",
    "status": "Waiting for data...",
    "prediction": "—",
    "prediction_class": "neutral"
}

lock = threading.Lock()

def classify_prediction(label: str) -> str:
    label = str(label).lower()
    if "high" in label:
        return "high"
    if "low" in label:
        return "low"
    return "neutral"

def compute_bmi(weight, height):
    return weight / ((height/100)**2)

def predict_from_inputs(age, sys_bp, dia_bp, bpm, temp, bmi, stress):
    values = np.array([[age, sys_bp, dia_bp, bpm, temp, bmi, stress]])
    scaled = scaler.transform(values)
    pred = model.predict(scaled)[0]
    return le.inverse_transform([pred])[0]

@app.route("/update")
def update():
    try:
        with lock:
            state["temp"] = float(request.args.get("temp"))
            state["bpm"] = float(request.args.get("bpm"))
            state["stress"] = int(float(request.args.get("stress")))
            state["spo2"] = float(request.args.get("spo2"))
            state["kicks"] = int(request.args.get("kicks"))
            state["timestamp"] = time.strftime("%H:%M:%S")
            state["status"] = "Live data received"

        return "OK"
    except Exception as e:
        return str(e)

@app.route("/")
def home():
    return render_template("index.html", sensor=state)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = float(request.form["age"])
        weight = float(request.form["weight"])
        height = float(request.form["height"])
        sys_bp = float(request.form["sys_bp"])
        dia_bp = float(request.form["dia_bp"])

        bmi = compute_bmi(weight, height)

        temp = state["temp"]
        bpm = state["bpm"]
        stress = state["stress"]

        result = predict_from_inputs(age, sys_bp, dia_bp, bpm, temp, bmi, stress)

        # 🔥 fallback logic
        if stress == 1 or sys_bp > 140 or dia_bp > 90:
            result = "High Risk"
        elif stress == 0 and sys_bp < 120 and dia_bp < 80:
            result = "Low Risk"
        else:
            result = "Moderate Risk"

        state["prediction"] = result
        state["prediction_class"] = classify_prediction(result)

        return render_template("index.html", sensor=state)

    except Exception as e:
        return str(e)

@app.route("/live")
def live():
    return jsonify(state)

if __name__ == "__main__":
    app.run(debug=True)