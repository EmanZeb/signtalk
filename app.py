"""Web app: python app.py  ->  open http://127.0.0.1:5000"""
import os, time, threading
import cv2, numpy as np
from flask import Flask, Response, jsonify, render_template, request
from engine import SignEngine
from gestures import GUIDE

app = Flask(__name__)
engine = SignEngine(hold_seconds=float(os.environ.get("HOLD", 1.0)))
CAM = int(os.environ.get("CAM", 0))
lock = threading.Lock()
latest = {"jpg": None, "error": None}


def placeholder(msg):
    img = np.full((480, 640, 3), 30, np.uint8)
    cv2.putText(img, msg, (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return cv2.imencode(".jpg", img)[1].tobytes()


def camera_loop():
    cap = cv2.VideoCapture(CAM)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        latest["jpg"] = placeholder("Camera not found (set CAM=1 ?)")
        latest["error"] = "Camera not found"
        return
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05); continue
        frame = cv2.flip(frame, 1)
        with lock:
            frame = engine.process(frame)
        latest["jpg"] = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


def stream():
    while True:
        if latest["jpg"]:
            yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + latest["jpg"] + b"\r\n"
        time.sleep(0.03)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=f")

@app.route("/api/state")
def state():
    with lock:
        s = engine.state()
    s["error"] = latest["error"]
    return jsonify(s)

@app.route("/api/guide")
def guide():
    return jsonify([{"word": w, "how": h} for w, h in GUIDE])

@app.route("/api/action", methods=["POST"])
def action():
    d = request.get_json(force=True)
    with lock:
        a = d.get("action")
        if a == "clear": engine.clear()
        elif a == "undo": engine.undo()
        elif a == "add" and d.get("word"): engine.add(str(d["word"])[:40])
    return jsonify(ok=True)


if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    print("Open http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)
