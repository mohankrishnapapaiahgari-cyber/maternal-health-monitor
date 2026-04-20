import serial
import requests

ser = serial.Serial('COM10', 9600)  # change if needed

while True:
    line = ser.readline().decode().strip()
    print("Raw:", line)

    try:
        temp, bpm, stress = line.split(',')

        url = f"https://maternal-health-monitor.onrender.com/update?temp={temp}&bpm={bpm}&stress={stress}"

        response = requests.get(url)
        print("Sent:", response.status_code)

    except:
        print("Error parsing:", line)