import serial
import requests

ser = serial.Serial("COM10", 9600)  # change COM if needed

URL = "https://maternal-health-monitor.onrender.com/update"

while True:
    line = ser.readline().decode().strip()
    print("Raw:", line)

    try:
        temp, bpm, stress, spo2, kicks = line.split(",")

        params = {
            "temp": temp,
            "bpm": bpm,
            "stress": stress,
            "spo2": spo2,
            "kicks": kicks
        }

        res = requests.get(URL, params=params)
        print("Sent:", res.status_code)

    except Exception as e:
        print("Error:", e)