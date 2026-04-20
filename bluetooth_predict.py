import os
import time
import serial
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "maternal_health_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "maternal_health_scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE_DIR, "maternal_health_labels.pkl"), "rb") as f:
    le = pickle.load(f)

PORTS_TO_TRY = ["COM7", "COM8", "COM9", "COM10"]

def open_bluetooth_port():
    while True:
        for port in PORTS_TO_TRY:
            try:
                print(f"Trying {port}...")
                ser = serial.Serial(port, 9600, timeout=2)
                time.sleep(2)
                ser.reset_input_buffer()
                print(f"Connected on {port}")
                return ser
            except serial.SerialException:
                pass
            except OSError:
                pass

        print("No Bluetooth port opened. Recheck pairing, then retrying in 5 seconds...")
        time.sleep(5)

ser = open_bluetooth_port()

print("Waiting for data from Arduino...")

while True:
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()

        if not line:
            continue

        print("Received:", line)

        values = [float(x) for x in line.split(",")]
        if len(values) != 7:
            print("Skipping invalid packet:", line)
            continue

        arr = np.array(values).reshape(1, -1)
        arr_scaled = scaler.transform(arr)
        pred = model.predict(arr_scaled)
        result = le.inverse_transform(pred)[0]

        print("Prediction:", result)
        print("---------------------------")

    except Exception as e:
        print("Error:", e)