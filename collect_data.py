"""Record your own signs (e.g. ASL letters).  python collect_data.py A B C
Hold SPACE to record samples for the current label, N = next label, Q = quit."""
import sys, os, csv, cv2, numpy as np, mediapipe as mp
from features import extract_features

labels = sys.argv[1:] or ["Hello", "Thanks", "Please"]
os.makedirs("data", exist_ok=True)
path = "data/samples.csv"
mph = mp.solutions.hands
hands = mph.Hands(max_num_hands=1, min_detection_confidence=0.6)
cap = cv2.VideoCapture(0)
i, count = 0, {l: 0 for l in labels}
with open(path, "a", newline="") as f:
    w = csv.writer(f)
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        k = cv2.waitKey(1) & 0xFF
        if res.multi_hand_landmarks:
            hl = res.multi_hand_landmarks[0]
            mp.solutions.drawing_utils.draw_landmarks(frame, hl, mph.HAND_CONNECTIONS)
            if k == 32:
                pts = np.array([[p.x, p.y, p.z] for p in hl.landmark])
                w.writerow([labels[i], *extract_features(pts)]); count[labels[i]] += 1
        cv2.putText(frame, f"Label: {labels[i]}  samples: {count[labels[i]]}  [SPACE=record N=next Q=quit]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Collect", frame)
        if k == ord("n"): i = (i + 1) % len(labels)
        if k == ord("q"): break
cap.release(); cv2.destroyAllWindows()
print("Saved to", path, count)
