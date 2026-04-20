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


def predict_from_inputs(age, bpm, temp, sys_bp, dia_bp, stress, weight):
    values = np.array([[age, bpm, temp, sys_bp, dia_bp, stress, weight]], dtype=float)
    scaled = scaler.transform(values)
    pred = model.predict(scaled)[0]

    if isinstance(pred, (int, np.integer)):
        return le.inverse_transform([int(pred)])[0]
    return str(pred)


# 🔴 NEW API (IMPORTANT)
@app.route("/update", methods=["GET"])
def update():
    global state

    try:
        temp = float(request.args.get("temp"))
        bpm = float(request.args.get("bpm"))
        stress = int(float(request.args.get("stress")))

        with lock:
            state["temp"] = temp
            state["bpm"] = bpm
            state["stress"] = stress
            state["timestamp"] = time.strftime("%H:%M:%S")
            state["status"] = "Live data received from device"

        return "OK"

    except Exception as e:
        return f"Error: {e}"


@app.route("/")
def home():
    with lock:
        sensor = state.copy()

    return render_template(
        "index.html",
        sensor=sensor,
        prediction=sensor["prediction"],
        age="",
        sys_bp="",
        dia_bp="",
        weight=""
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = float(request.form["age"])
        sys_bp = float(request.form["sys_bp"])
        dia_bp = float(request.form["dia_bp"])
        weight = float(request.form["weight"])

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
                sys_bp=request.form.get("sys_bp", ""),
                dia_bp=request.form.get("dia_bp", ""),
                weight=request.form.get("weight", "")
            )

        result = predict_from_inputs(age, bpm, temp, sys_bp, dia_bp, stress, weight)
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
            sys_bp=sys_bp,
            dia_bp=dia_bp,
            weight=weight
        )

    except Exception as e:
        with lock:
            sensor = state.copy()

        return render_template(
            "index.html",
            sensor=sensor,
            prediction=f"Error: {e}",
            age=request.form.get("age", ""),
            sys_bp=request.form.get("sys_bp", ""),
            dia_bp=request.form.get("dia_bp", ""),
            weight=request.form.get("weight", "")
        )


@app.route("/live")
def live():
    with lock:
        return jsonify(state)


if __name__ == "__main__":
    app.run(debug=True)