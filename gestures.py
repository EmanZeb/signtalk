"""Rule-based static hand-gesture classifier working on MediaPipe's 21 hand landmarks.
Works out-of-the-box (no training needed). Rotation-invariant finger-state detection."""
import numpy as np

# (thumb, index, middle, ring, pinky) -> meaning
GESTURES = {
    (1, 1, 1, 1, 1): "Hello",
    (0, 1, 1, 1, 1): "Stop",
    (0, 1, 1, 0, 0): "Peace",
    (0, 1, 0, 0, 0): "You",
    (0, 0, 0, 0, 1): "I",
    (0, 1, 1, 1, 0): "Water",
    (1, 1, 0, 0, 1): "I love you",
    (1, 0, 0, 0, 1): "Call me",
    (1, 1, 0, 0, 0): "Help",
    (1, 1, 1, 0, 0): "Thank you",
    (0, 0, 0, 0, 0): "Yes",
}

GUIDE = [
    ("Hello", "Open palm, all 5 fingers spread"),
    ("Stop", "4 fingers up, thumb folded in"),
    ("Peace", "Index + middle finger up (V sign)"),
    ("You", "Only index finger pointing up"),
    ("I", "Only pinky finger up"),
    ("Water", "Index, middle, ring up (W shape)"),
    ("I love you", "Thumb + index + pinky out"),
    ("Call me", "Thumb + pinky out (phone sign)"),
    ("Help", "Thumb + index out (L shape)"),
    ("Thank you", "Thumb + index + middle out"),
    ("Yes", "Closed fist"),
    ("Good", "Thumbs up"),
    ("No", "Thumbs down"),
    ("OK", "Thumb & index touch in a circle, other 3 up"),
]


def _d(a, b):
    return float(np.linalg.norm(np.asarray(a)[:2] - np.asarray(b)[:2]))


def finger_states(lm):
    """Return tuple of 5 ints (1 = extended) and palm size."""
    lm = np.asarray(lm)[:, :2]
    palm = _d(lm[0], lm[9]) + 1e-6
    thumb = int(_d(lm[4], lm[17]) > _d(lm[3], lm[17]) * 1.05 and _d(lm[4], lm[5]) > 0.55 * palm)
    others = []
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        others.append(int(_d(lm[tip], lm[0]) > _d(lm[pip], lm[0]) * 1.05))
    return (thumb, *others), palm


def classify_rules(lm):
    """lm: (21,3) array of normalised landmarks. Returns label or None."""
    lm = np.asarray(lm)
    fs, palm = finger_states(lm)
    thumb, idx, mid, ring, pinky = fs

    # OK sign: thumb tip touches index tip, other three fingers extended
    if _d(lm[4], lm[8]) < 0.35 * palm and mid and ring and pinky:
        return "OK"

    # Fist-like hand (other 4 fingers folded): thumb up / down / plain fist
    if not (idx or mid or ring or pinky):
        dy = lm[2][1] - lm[4][1]  # image y grows downward
        if dy > 0.45 * palm:
            return "Good"
        if dy < -0.45 * palm:
            return "No"
        if thumb:
            return None  # thumb sticking out sideways: ambiguous
        return "Yes"

    return GESTURES.get(fs)
