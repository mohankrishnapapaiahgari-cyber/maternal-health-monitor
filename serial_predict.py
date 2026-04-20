import serial
import pickle
import numpy as np
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "maternal_health_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "maternal_health_scaler.pkl"), "rb"))
le = pickle.load(open(os.path.join(BASE_DIR, "maternal_health_labels.pkl"), "rb"))

ser = serial.Serial('COM10', 9600, timeout=2)
time.sleep(2)
ser.reset_input_buffer()

print("Connected via USB. Waiting for data...")

while True:
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        line = line.replace("\x00", "")

        if not line:
            continue

        parts = line.split(",")
        if len(parts) != 7:
            continue

        values = [float(x) for x in parts]
        print("Received:", line)

        arr = np.array(values).reshape(1, -1)
        arr_scaled = scaler.transform(arr)
        pred = model.predict(arr_scaled)
        result = le.inverse_transform(pred)[0]

        print("Prediction:", result)
        print("------------------")

    except Exception as e:
        print("Error:", e)