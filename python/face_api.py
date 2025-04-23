from flask import Flask, request, jsonify, render_template_string
import cv2
import numpy as np
import face_recognition
import pickle
import base64
import os
import csv
from datetime import datetime

app = Flask(__name__)

# Load face encodings
with open(
    "C:/Users/raoul/Downloads/capstone/capstone/python-api/encodings/face_encodings.pkl",
    "rb",
) as f:
    encodeListKnown, classNames = pickle.load(f)

# Attendance file path
ATTENDANCE_FILE = "attendance.csv"

# Function to mark attendance
def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Create file if not exists
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])

    # Check if already marked today
    already_marked = set()
    with open(ATTENDANCE_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Name"] == name and row["Date"] == date_str:
                already_marked.add(name)

    if name not in already_marked:
        with open(ATTENDANCE_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, date_str, time_str])


# Home page to show webcam interface
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Real-Time Face Recognition</title>
</head>
<body>
    <h2>Webcam Face Recognition</h2>
    <video id="video" width="480" height="360" autoplay></video>
    <p id="result">Result: Waiting...</p>

    <script>
        const video = document.getElementById('video');
        const resultText = document.getElementById('result');

        // Access webcam
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => { video.srcObject = stream; });

        // Capture and send frame every 1 second
        setInterval(() => {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);
            const imageData = canvas.toDataURL('image/jpeg');

            fetch('/api/recognize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            })
            .then(res => res.json())
            .then(data => {
                if (data.results.length > 0) {
                    resultText.textContent = "Result: " + data.results[0].name + " (Confidence: " + data.results[0].confidence.toFixed(2) + ")";
                } else {
                    resultText.textContent = "Result: No match";
                }
            });
        }, 1000);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/api/recognize", methods=["POST"])
def recognize():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image received"}), 400

    # Decode base64 image
    image_data = data["image"].split(",")[1]
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Resize + Convert
    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Face detection
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    results = []
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            confidence = 1 - faceDis[matchIndex]
            results.append(
                {"name": name, "confidence": float(confidence), "location": faceLoc}
            )
            mark_attendance(name)  # ✅ Mark attendance here

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(debug=True)
