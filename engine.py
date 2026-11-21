"""Core engine: camera frame -> hand landmarks -> gesture -> word -> sentence."""
import os, time, collections
import cv2
import numpy as np
import mediapipe as mp
from gestures import classify_rules
from features import extract_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "sign_model.pkl")


class SignEngine:
    def __init__(self, hold_seconds=1.0, min_conf=0.70):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=1, model_complexity=1,
                                         min_detection_confidence=0.6,
                                         min_tracking_confidence=0.5)
        self.hold = hold_seconds
        self.min_conf = min_conf
        self.model = None
        if os.path.exists(MODEL_PATH):
            import joblib
            self.model = joblib.load(MODEL_PATH)
        self.history = collections.deque(maxlen=6)
        self.candidate, self.cand_start, self.locked = None, 0.0, False
        self.sentence = []
        self.version = 0          # increments each time a word is added
        self.current, self.confidence, self.progress = None, 0.0, 0.0

    # ---- recognition -------------------------------------------------
    @property
    def mode(self):
        return "Custom ML model" if self.model is not None else "Built-in gestures"

    def _predict(self, pts):
        if self.model is not None:
            f = extract_features(pts).reshape(1, -1)
            proba = self.model.predict_proba(f)[0]
            i = int(np.argmax(proba))
            if proba[i] >= self.min_conf:
                return str(self.model.classes_[i]), float(proba[i])
            return None, float(proba[i])
        label = classify_rules(pts)
        return label, (0.95 if label else 0.0)

    def _smooth(self, label):
        self.history.append(label)
        if label is None:
            return None
        return label if self.history.count(label) >= 3 else self.current

    def _commit_logic(self, label, now):
        if label is None:
            self.candidate, self.locked, self.progress = None, False, 0.0
            return
        if label != self.candidate:
            self.candidate, self.cand_start, self.locked, self.progress = label, now, False, 0.0
            return
        if self.locked:
            self.progress = 1.0
            return
        self.progress = min(1.0, (now - self.cand_start) / self.hold)
        if self.progress >= 1.0:
            self.sentence.append(label)
            self.version += 1
            self.locked = True

    # ---- per-frame ---------------------------------------------------
    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        label, conf = None, 0.0
        if res.multi_hand_landmarks:
            hl = res.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame_bgr, hl, self.mp_hands.HAND_CONNECTIONS)
            pts = np.array([[p.x, p.y, p.z] for p in hl.landmark])
            label, conf = self._predict(pts)
        else:
            self.history.append(None)
        label = self._smooth(label) if res.multi_hand_landmarks else None
        self.current, self.confidence = label, conf
        self._commit_logic(label, time.time())
        self._overlay(frame_bgr)
        return frame_bgr

    def _overlay(self, img):
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 48), (30, 30, 30), -1)
        txt = self.current if self.current else "Show a sign..."
        cv2.putText(img, txt, (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 170), 2)
        cv2.rectangle(img, (0, 48), (int(w * self.progress), 56), (0, 200, 255), -1)
        cv2.rectangle(img, (0, h - 40), (w, h), (30, 30, 30), -1)
        cv2.putText(img, " ".join(self.sentence)[-60:], (12, h - 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # ---- sentence helpers -------------------------------------------
    def clear(self):
        self.sentence = []; self.version += 1

    def undo(self):
        if self.sentence: self.sentence.pop(); self.version += 1

    def add(self, word):
        self.sentence.append(word); self.version += 1

    def state(self):
        return {"current": self.current, "confidence": round(self.confidence, 2),
                "progress": round(self.progress, 2), "sentence": self.sentence,
                "version": self.version, "mode": self.mode}
