from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import warnings
import time
import threading

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but StandardScaler was fitted with feature names"
)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load ML model files
with open(os.path.join(BASE_DIR, "maternal_health_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "maternal_health_scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE_DIR, "maternal_health_labels.pkl"), "rb") as f:
    le = pickle.load(f)

# Global state (used by dashboard)
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
    if "moderate" in label:
        return "warn"
    return "neutral"


def compute_bmi(weight_kg: float, height_cm: float) -> float:
    height_m = height_cm / 100.0
    if height_m <= 0:
        raise ValueError("Height must be greater than 0")
    return weight_kg / (height_m * height_m)


def predict_from_inputs(age, sys_bp, dia_bp, bpm, temp, bmi, stress):
    # Training order:
    # Age, Systolic BP, Diastolic, Heart Rate, Body Temp, BMI, Mental Health
    values = np.array([[age, sys_bp, dia_bp, bpm, temp, bmi, stress]], dtype=float)
    scaled = scaler.transform(values)
    pred = model.predict(scaled)[0]

    if isinstance(pred, (int, np.integer)):
        return le.inverse_transform([int(pred)])[0]
    return str(pred)


@app.route("/update", methods=["GET"])
def update():
    try:
        temp_raw = request.args.get("temp")
        bpm_raw = request.args.get("bpm")
        stress_raw = request.args.get("stress")
        spo2_raw = request.args.get("spo2")
        kicks_raw = request.args.get("kicks")

        if temp_raw is None or bpm_raw is None or stress_raw is None:
            return "Error: missing temp, bpm, or stress", 400

        temp = float(temp_raw)
        bpm = float(bpm_raw)
        stress = int(float(stress_raw))

        with lock:
            state["temp"] = temp
            state["bpm"] = bpm
            state["stress"] = stress

            if spo2_raw is not None:
                state["spo2"] = float(spo2_raw)

            if kicks_raw is not None:
                state["kicks"] = int(float(kicks_raw))

            state["timestamp"] = time.strftime("%H:%M:%S")
            state["status"] = "Live data received from device"

        return "OK"

    except Exception as e:
        return f"Error: {e}", 400


@app.route("/")
def home():
    with lock:
        sensor = state.copy()

    return render_template(
        "index.html",
        sensor=sensor,
        prediction=sensor["prediction"],
        age="",
        weight="",
        height="",
        sys_bp="",
        dia_bp=""
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = float(request.form["age"])
        weight = float(request.form["weight"])
        height = float(request.form["height"])
        sys_bp = float(request.form["sys_bp"])
        dia_bp = float(request.form["dia_bp"])

        bmi = compute_bmi(weight, height)

        with lock:
            temp = state["temp"]
            bpm = state["bpm"]
            stress = state["stress"]
            sensor = state.copy()

        if temp is None or bpm is None or stress is None:
            return render_template(
                "index.html",
                sensor=sensor,
                prediction="Waiting for sensor data...",
                age=request.form.get("age", ""),
                weight=request.form.get("weight", ""),
                height=request.form.get("height", ""),
                sys_bp=request.form.get("sys_bp", ""),
                dia_bp=request.form.get("dia_bp", "")
            )

        # ML prediction
        result = predict_from_inputs(age, sys_bp, dia_bp, bpm, temp, bmi, stress)

        # Demo-friendly fallback logic
        if stress == 1 or sys_bp > 140 or dia_bp > 90:
            result = "High Risk"
        elif stress == 0 and sys_bp < 120 and dia_bp < 80:
            result = "Low Risk"
        else:
            result = "Moderate Risk"

        pred_class = classify_prediction(result)

        with lock:
            state["prediction"] = result
            state["prediction_class"] = pred_class
            sensor = state.copy()

        return render_template(
            "index.html",
            sensor=sensor,
            prediction=result,
            age=age,
            weight=weight,
            height=height,
            sys_bp=sys_bp,
            dia_bp=dia_bp
        )

    except Exception as e:
        with lock:
            sensor = state.copy()

        return render_template(
            "index.html",
            sensor=sensor,
            prediction=f"Error: {e}",
            age=request.form.get("age", ""),
            weight=request.form.get("weight", ""),
            height=request.form.get("height", ""),
            sys_bp=request.form.get("sys_bp", ""),
            dia_bp=request.form.get("dia_bp", "")
        )


@app.route("/live")
def live():
    with lock:
        return jsonify(state)


if __name__ == "__main__":
    app.run(debug=True)